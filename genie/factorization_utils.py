import torch
import torch.nn as nn
from einops import rearrange


class FactorizedEmbedding(nn.Module):
    """
    Each token's embedding is the sum of the embeddings in each factorized vocabulary.
    Equivalent to nn.Embedding when `num_factored_vocabs` = 1.
    """
    def __init__(self, factored_vocab_size: int, num_factored_vocabs: int, d_model: int, mask_token_id: int):
        """

        Args:
            config: Should specify `factored_vocab_size`, `d_model`, `num_factored_vocabs`, `image_vocab_size`.
                E.g. genie.config.GenieConfig
        """
        super().__init__()

        self.factored_vocab_size = factored_vocab_size
        self.num_factored_vocabs = num_factored_vocabs
        self.d_model = d_model
        self.mask_token_id = mask_token_id

        self.factored_embeds = nn.ParameterList([nn.Embedding(factored_vocab_size, d_model)
                                                 for _ in range(num_factored_vocabs)])
        self.mask_token_embed = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """

        Args:
            input_ids: Shape (B, T, H*W)

        Returns:
            input embeddings: Shape (B, T, H*W, d_model)
        """
        # initialize all embeddings to the mask token embedding, and then fill in actual token embeddings
        embeds = self.mask_token_embed.repeat(input_ids.size() + (1,))
        is_not_mask = input_ids != self.mask_token_id

        factored_token_ids = factorize_token_ids(
            input_ids[is_not_mask], self.num_factored_vocabs, self.factored_vocab_size
        )

        unmasked_embeds = [
            factored_embed(factored_token_ids)
            for factored_embed, factored_token_ids in zip(self.factored_embeds, factored_token_ids.unbind(-1))
        ]

        embeds[is_not_mask] = torch.sum(torch.stack(unmasked_embeds), dim=0)
        return embeds


def factorize_token_ids(
    token_ids: torch.LongTensor,
    num_factored_vocabs: int = 2,
    factored_vocab_size: int = 512
) -> torch.LongTensor:
    """
    `token_ids`: any size tensor with token id values in [0, image_vocab_size = 2**18).

    Returns:
        Size token_ids.size() + (num_factored_vocabs,), where the last dimension has token ids in
        each individual vocabulary, with values in [0, factored_vocab_size = 512)
    """
    powers = factored_vocab_size ** torch.arange(num_factored_vocabs, device=token_ids.device)
    return (token_ids.unsqueeze(-1) // powers) % factored_vocab_size


def unfactorize_token_ids(
    factored_token_ids: torch.LongTensor,
    num_factored_vocabs: int = 2,
    factored_vocab_size: int = 512
) -> torch.LongTensor:
    """
    Inverse of `factorize_token_ids`.
    It is assumed that the last dimension of `factored_token_ids` is the vocabulary dimension.

    Returns:
        Size token_ids.size()[:-1], with values in [0, image_vocab_size = 2**18)
    """
    powers = factored_vocab_size ** torch.arange(num_factored_vocabs, device=factored_token_ids.device)
    return (factored_token_ids * powers).sum(dim=-1)


def factorize_labels(
    labels_THW: torch.LongTensor,
    num_factored_vocabs: int = 2,
    factored_vocab_size: int = 512
) -> torch.LongTensor:
    """
    Simply `factorize_token_ids` followed by permuting dimensions.
    labels_THW: shape (B, T, H, W), values in [0, image_vocab_size=2**18)

    Returns:
        factored_labels: shape (B, num_factored_vocabs=2, T, H, W), values in [0, factored_vocab_size=512)
    """
    factored_labels = factorize_token_ids(labels_THW, num_factored_vocabs, factored_vocab_size)
    return rearrange(factored_labels, "b t h w num_factored_vocabs -> b num_factored_vocabs t h w")


def nth_root(x, n):
    root = round(x ** (1 / n))
    assert root ** n == x, (x, n, root)
    return root
