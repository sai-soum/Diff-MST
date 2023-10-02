import torch
import laion_clap


class StereoCLAPLoss(torch.nn.Module):
    def __init__(self, sum_and_diff: bool = False, distance: str = "l2"):
        super().__init__()
        self.sum_and_diff = sum_and_diff
        self.distance = distance

        # instatiate pretrained CLAP model
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()  # download the default pretrained checkpoint.

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Compute loss on stereo mixes using featues from pretrained CLAP model.

        Args:
            input: (bs, 2, seq_len)
            target: (bs, 2, seq_len)

        Returns:
            loss: (batch_size, )
        """
        bs, chs, seq_len = input.size()

        assert chs == 2, "Input must be stereo"

        if self.sum_and_diff:
            # compute sum and diff of stereo channels
            input_sum = input[:, 0, :] + input[:, 1, :]
            input_diff = input[:, 0, :] - input[:, 1, :]
            target_sum = target[:, 0, :] + target[:, 1, :]
            target_diff = target[:, 0, :] - target[:, 1, :]

            # compute embeddings
            input_sum_embeddings = self.model.get_audio_embedding_from_data(
                x=input_sum, use_tensor=True
            )
            target_sum_embeddings = self.model.get_audio_embedding_from_data(
                x=target_sum, use_tensor=True
            )
            input_diff_embeddings = self.model.get_audio_embedding_from_data(
                x=input_diff, use_tensor=True
            )
            target_diff_embeddings = self.model.get_audio_embedding_from_data(
                x=target_diff, use_tensor=True
            )

            # compute losses
            if self.distance == "l2":
                sum_loss = torch.nn.functional.mse_loss(
                    input_sum_embeddings, target_sum_embeddings
                )
                diff_loss = torch.nn.functional.mse_loss(
                    input_diff_embeddings, target_diff_embeddings
                )
            elif self.distance == "l1":
                sum_loss = torch.nn.functional.l1_loss(
                    input_sum_embeddings, target_sum_embeddings
                )
                diff_loss = torch.nn.functional.l1_loss(
                    input_diff_embeddings, target_diff_embeddings
                )
            else:
                raise ValueError(f"Invalid distance {self.distance}")

            # compute total loss
            loss = (sum_loss + diff_loss) / 2

        else:
            # move channel dim to batch dim
            input = input.view(bs * 2, -1)
            target = target.view(bs * 2, -1)

            # compute embeddings
            input_embeddings = self.model.get_audio_embedding_from_data(
                x=input, use_tensor=True
            )
            target_embeddings = self.model.get_audio_embedding_from_data(
                x=target, use_tensor=True
            )

            # compute losses
            if self.distance == "l2":
                loss = torch.nn.functional.mse_loss(input_embeddings, target_embeddings)
            elif self.distance == "l1":
                loss = torch.nn.functional.l1_loss(input_embeddings, target_embeddings)
            else:
                raise ValueError(f"Invalid distance {self.distance}")

        return loss
