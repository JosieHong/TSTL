import torch
import torch.nn as nn
import torch.nn.functional as F
from decimal import *
from typing import Tuple



# ----------------------------------------
# >>>           encoder part           <<<
# ----------------------------------------
class MolConv3(nn.Module):
	def __init__(self, in_dim, out_dim, point_num, k, remove_xyz=False):
		super(MolConv3, self).__init__()
		self.k = k
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.remove_xyz = remove_xyz

		self.dist_ff = nn.Sequential(
			nn.Conv2d(1, 1, kernel_size=1, bias=False), 
			nn.LayerNorm((1, point_num, k)),
			nn.Sigmoid()
		)

		if remove_xyz:
			self.center_ff = nn.Sequential(
				nn.Conv2d(in_dim - 3, in_dim + k - 3, kernel_size=1, bias=False), 
				nn.LayerNorm((in_dim + k - 3, point_num, k)),
				nn.Sigmoid(), 
			)
			self.update_ff = nn.Sequential(
				nn.Conv2d(in_dim + k - 3, out_dim, kernel_size=1, bias=False), 
				nn.LayerNorm((out_dim, point_num, k)),  
				nn.Softplus(beta=1.0, threshold=20.0), 
			)
		else:
			self.center_ff = nn.Sequential(
				nn.Conv2d(in_dim, in_dim + k, kernel_size=1, bias=False), 
				nn.LayerNorm((in_dim + k, point_num, k)),
				nn.Sigmoid()
			)
			self.update_ff = nn.Sequential(
				nn.Conv2d(in_dim + k, out_dim, kernel_size=1, bias=False), 
				nn.LayerNorm((out_dim, point_num, k)), 
				nn.Softplus(beta=1.0, threshold=20.0), 
			)

	def forward(self, x: torch.Tensor, idx_base: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: 
		# Generate features
		dist, gm2, feat_c, feat_n = self._generate_feat(x, idx_base, k=self.k, remove_xyz=self.remove_xyz)
		'''Returned features: 
		dist: torch.Size([batch_size, 1, point_num, k])
		gm2: torch.Size([batch_size, k, point_num, k]) 
		feat_c: torch.Size([batch_size, in_dim, point_num, k]) 
		feat_n: torch.Size([batch_size, in_dim, point_num, k])
		'''
		feat_n = torch.cat((feat_n, gm2), dim=1) # torch.Size([batch_size, in_dim+k, point_num, k])
		feat_c = self.center_ff(feat_c)
		w = self.dist_ff(dist)
	
		feat = w * feat_n + feat_c
		feat = self.update_ff(feat)

		# Average pooling along the fourth dimension
		mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand_as(feat) # [batch_size, out_dim, point_num, k]
		feat = feat.masked_fill(~mask_expanded, 0.0) # Set padding points to zero
		valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=0.1) # Avoid division by zero
		feat = feat.sum(dim=3) / valid_counts.unsqueeze(2) # [batch_size, out_dim, point_num]
		return feat

	def _generate_feat(self, x: torch.Tensor, 
								idx_base: torch.Tensor, 
								k: int, 
								remove_xyz: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
		batch_size, num_dims, num_points = x.size()
		
		# local graph (knn)
		inner = -2*torch.matmul(x.transpose(2, 1), x)
		xx = torch.sum(x**2, dim=1, keepdim=True)
		pairwise_distance = -xx - inner - xx.transpose(2, 1)
		dist, idx = pairwise_distance.topk(k=k, dim=2) # (batch_size, num_points, k)
		dist = - dist
		
		idx = idx + idx_base
		idx = idx.view(-1)

		x = x.transpose(2, 1).contiguous() # (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims) 
		# print('_double_gram_matrix (x):', torch.any(torch.isnan(x)))
		graph_feat = x.view(batch_size*num_points, -1)[idx, :]
		# print('_double_gram_matrix (graph_feat):', torch.any(torch.isnan(graph_feat)))
		graph_feat = graph_feat.view(batch_size, num_points, k, num_dims)
		
		# gram matrix
		gm_matrix = torch.matmul(graph_feat, graph_feat.permute(0, 1, 3, 2))
		gm_matrix = F.normalize(gm_matrix, dim=1)
		# print('_double_gram_matrix (gm_matrix):', torch.any(torch.isnan(gm_matrix)))

		# double gram matrix
		sub_feat = gm_matrix[:, :, :, 0].unsqueeze(3)
		sub_gm_matrix = torch.matmul(sub_feat, sub_feat.permute(0, 1, 3, 2))
		sub_gm_matrix = F.normalize(sub_gm_matrix, dim=1)
		# print('_double_gram_matrix (sub_gm_matrix):', torch.any(torch.isnan(sub_gm_matrix)))

		x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
		
		if remove_xyz:
			dist = dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous()
			gm2 = sub_gm_matrix.permute(0, 3, 1, 2).contiguous() 
			feat_c = x[:, :, :, 3:].permute(0, 3, 1, 2).contiguous() 
			feat_n = graph_feat[:, :, :, 3:].permute(0, 3, 1, 2).contiguous()
		else:
			dist = dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous()
			gm2 = sub_gm_matrix.permute(0, 3, 1, 2).contiguous()
			feat_c = x.permute(0, 3, 1, 2).contiguous()
			feat_n = graph_feat.permute(0, 3, 1, 2).contiguous()

		return dist, gm2, feat_c, feat_n
	
	def __repr__(self):
		return self.__class__.__name__ + ' k = ' + str(self.k) + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'

class Encoder(nn.Module):
	def __init__(self, in_dim, layers, emb_dim, point_num, k): 
		super(Encoder, self).__init__()
		self.emb_dim = emb_dim 
		self.hidden_layers = nn.ModuleList([MolConv3(in_dim=in_dim, out_dim=layers[0], point_num=point_num, k=k, remove_xyz=True)])
		for i in range(1, len(layers)): 
			if i == 1:
				self.hidden_layers.append(MolConv3(in_dim=layers[i-1], out_dim=layers[i], point_num=point_num, k=k, remove_xyz=False))
			else:
				self.hidden_layers.append(MolConv3(in_dim=layers[i-1], out_dim=layers[i], point_num=point_num, k=k, remove_xyz=False))
		
		self.conv = nn.Sequential(nn.Conv1d(emb_dim, emb_dim, kernel_size=1, bias=False), 
								nn.LayerNorm((emb_dim, point_num)), 
								nn.LeakyReLU(negative_slope=0.2))

	def forward(self, x: torch.Tensor,  
						idx_base: torch.Tensor,
						mask: torch.Tensor) -> torch.Tensor: 
		xs = []
		for i, hidden_layer in enumerate(self.hidden_layers): 
			if i == 0: 
				tmp_x = hidden_layer(x, idx_base, mask)
			else: 
				tmp_x = hidden_layer(xs[-1], idx_base, mask)
			xs.append(tmp_x)

		x = torch.cat(xs, dim=1) # torch.Size([batch_size, emb_dim, point_num])
		x = self.conv(x)
		
		# Apply the mask: Set padding points to a very low value for max pooling and zero for average pooling
		mask_expanded = mask.unsqueeze(1).expand_as(x) # [batch_size, emb_dim, point_num]
		x_masked_max = x.masked_fill(~mask_expanded, float('-inf')) # Replace padding with -inf for max pooling
		x_masked_avg = x.masked_fill(~mask_expanded, 0.0) # Replace padding with 0 for average pooling
		
		# Max pooling along the third dimension
		max_pooled = torch.max(x_masked_max, dim=2)[0] # [batch_size, emb_dim]
		
		# Average pooling along the third dimension
		# Count the valid (non-padding) points for each position
		valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=0.1) # Avoid division by zero
		avg_pooled = x_masked_avg.sum(dim=2) / valid_counts # [batch_size, emb_dim]
		
		x = max_pooled + avg_pooled
		return x

# ----------------------------------------
# >>>           decoder part           <<<
# ----------------------------------------
class FCResBlock(nn.Module): 
	def __init__(self, in_dim: int, out_dim: int, dropout: float=0.) -> torch.Tensor: 
		super(FCResBlock, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim

		self.linear1 = nn.Linear(in_dim, out_dim, bias=False) 
		self.bn1 = nn.LayerNorm(out_dim)

		self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
		self.bn2 = nn.LayerNorm(out_dim)

		self.linear3 = nn.Linear(out_dim, out_dim, bias=False)
		self.bn3 = nn.LayerNorm(out_dim)

		self.dp = nn.Dropout(dropout)

	def reset_parameters(self):
		self.linear1.reset_parameters()
		self.linear2.reset_parameters()
		self.linear3.reset_parameters()
		self.bn1.reset_parameters()
		self.bn2.reset_parameters()
		self.bn3.reset_parameters()

	def forward(self, x):
		identity = x
		
		x = self.bn1(self.linear1(x))
		x = F.leaky_relu(x, negative_slope=0.2)
		x = self.bn2(self.linear2(x))
		x = F.leaky_relu(x, negative_slope=0.2)
		x = self.bn3(self.linear3(x))
		
		x = x + F.interpolate(identity.unsqueeze(1), size=x.size()[1]).squeeze()

		x = F.leaky_relu(x, negative_slope=0.2)
		x = self.dp(x)
		return x

	def __repr__(self):
		return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'

# -------------------------------------------------------------------------
# >>>                            Final model                            <<<
# -------------------------------------------------------------------------
class MolNet_RT(nn.Module): 
	def __init__(self, config): 
		super(MolNet_RT, self).__init__()
		self.add_num = config['add_num']
		self.encoder = Encoder(in_dim=int(config['in_dim']), 
									layers=config['encode_layers'],
									emb_dim=int(config['emb_dim']),
									point_num=int(config['max_atom_num']), 
									k=int(config['k']))
		self.decoder = nn.Sequential(
							FCResBlock(in_dim=int(config['emb_dim'] + config['add_num']), out_dim=config['decode_layers'][0]),
							*[FCResBlock(in_dim=config['decode_layers'][i], 
										out_dim=config['decode_layers'][i+1],
										dropout=config['dropout'] if len(config['decode_layers']) - i <= 3 else 0.0) 
							for i in range(len(config['decode_layers'])-1)],
							nn.Linear(config['decode_layers'][-1], config['out_dim'])
						)
		
		for m in self.modules():
			if isinstance(m, (nn.Conv1d, nn.Conv2d)): 
				nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
			elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)): 
				nn.init.constant_(m.weight, 1)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear): 
				nn.init.xavier_normal_(m.weight) 
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, x: torch.Tensor, 
						mask: torch.Tensor, 
						env: torch.Tensor = None, 
						idx_base: torch.Tensor = None) -> torch.Tensor: 
		'''
		Input: 
			x:      	point set, torch.Size([batch_size, num_dims, num_points])
			mask:		mask for padding points, torch.Size([batch_size, num_points])
			env:    	experimental condiction, torch.Size([batch_size, add_num])
			idx_base:	idx for local knn, torch.Size([batch_size, 1, 1])
		'''
		if idx_base is None:
			batch_size, num_dims, num_points = x.size()
			idx_base = torch.arange(0, batch_size, device=x.device, dtype=torch.long).view(-1, 1, 1) * num_points
		x = self.encoder(x, idx_base, mask) # torch.Size([batch_size, emb_dim])

		# add the encoded adduct
		if self.add_num == 1:
			x = torch.cat((x, torch.unsqueeze(env, 1)), 1)
		elif self.add_num > 1:
			x = torch.cat((x, env), 1)

		# decoder
		x = self.decoder(x)
		return x
	