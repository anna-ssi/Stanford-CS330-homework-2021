"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding_sharing = embedding_sharing

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        if embedding_sharing:
            self.user_embeds = ScaledEmbedding(
                num_users, self.embedding_dim, sparse=sparse)
            self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)

            self.item_embeds = ScaledEmbedding(
                num_items, self.embedding_dim, sparse=sparse)
            self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

            self.user_embeds.reset_parameters()
            self.user_biases.reset_parameters()
            self.item_embeds.reset_parameters()
            self.item_biases.reset_parameters()

        else:
            self.reg_user_embeds = ScaledEmbedding(
                num_users, self.embedding_dim, sparse=sparse)
            self.reg_user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)

            self.reg_item_embeds = ScaledEmbedding(
                num_items, self.embedding_dim, sparse=sparse)
            self.reg_item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

            self.fac_user_embeds = ScaledEmbedding(
                num_users, self.embedding_dim, sparse=sparse)
            self.fac_item_embeds = ScaledEmbedding(
                num_items, self.embedding_dim, sparse=sparse)

            self.fac_user_embeds.reset_parameters()
            self.fac_item_embeds.reset_parameters()

            self.reg_user_embeds.reset_parameters()
            self.reg_item_embeds.reset_parameters()
            self.reg_item_biases.reset_parameters()
            self.reg_user_biases.reset_parameters()

        self.model = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], 1)
        )

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of 
            shape (batch,). This corresponds to p_ij in the 
            assignment.
        score: tensor
            Tensor of user-item score predictions of shape 
            (batch,). This corresponds to r_ij in the 
            assignment.
        """

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        if self.embedding_sharing:
            user_embed = self.user_embeds(user_ids)
            user_bias = self.user_biases(user_ids)

            item_embed = self.item_embeds(item_ids)
            item_bias = self.item_biases(item_ids)

            predictions = user_embed @ item_embed.T + user_bias + item_bias

            final_embed = torch.cat(
                (user_embed, item_embed, user_embed * item_embed), 1)
            score = self.model(final_embed)

        else:
            fac_user_embed = self.fac_user_embeds(user_ids)
            fac_item_embed = self.fac_item_embeds(item_ids)

            final_embed = torch.cat(
                (fac_user_embed, fac_item_embed, fac_user_embed * fac_item_embed), 1)
            score = self.model(final_embed)

            reg_user_embed = self.reg_user_embeds(user_ids)
            reg_user_bias = self.reg_user_biases(user_ids)

            reg_item_embed = self.reg_item_embeds(item_ids)
            reg_item_bias = self.reg_item_biases(item_ids)

            predictions = reg_user_embed @ reg_item_embed.T + reg_user_bias + reg_item_bias

        # ********************************************************
        # ********************************************************
        # ********************************************************

        return predictions, score
