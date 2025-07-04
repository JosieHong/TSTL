# -*- coding: utf-8 -*-
#
# Modified version of pagtn_predictor.py
# Original version available at https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/model_zoo/mpnn_predictor.py
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MPNN
# pylint: disable= no-member, arguments-differ, invalid-name

import torch.nn as nn

from dgl.nn.pytorch import Set2Set

from .mpnn import MPNN

__all__ = ['MPNNPredictor']

# pylint: disable=W0221
class MPNNPredictor(nn.Module):
    """MPNN for regression and classification on graphs.

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_hidden_feats : int
        Size for the hidden edge representations. Default to 128.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    num_step_set2set : int
        Number of set2set steps. Default to 6.
    num_layer_set2set : int
        Number of set2set layers. Default to 3.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 dropout=0.1,
                 num_step_message_passing=5,
                 num_step_set2set=6,
                 num_layer_set2set=3,
                 n_tasks=1,
                 predictor_hidden_feats=256):
        super(MPNNPredictor, self).__init__()

        self.encoder = MPNN(node_in_feats=node_in_feats,
                        node_out_feats=node_out_feats,
                        edge_in_feats=edge_in_feats,
                        edge_hidden_feats=edge_hidden_feats,
                        num_step_message_passing=num_step_message_passing,
                        dropout=dropout)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
                               
        self.decoder = nn.Sequential(
            nn.Linear(2 * node_out_feats, predictor_hidden_feats), nn.ReLU(),
            nn.Linear(predictor_hidden_feats, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = self.encoder(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        output = self.decoder(graph_feats)

        return output
