from torch.optim import Adam, Adagrad


class OptimWarmupEncoderDecoder(object):
    """Custom wrapper for Adam optimizer, handles lr warmup and smaller lr for encoder fine-tuning."""

    def __init__(self, model, args):
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.lr = {
            "encoder": args.encoder_learning_rate,
            "decoder": args.decoder_learning_rate,
            "text_span_decoder": args.decoder_learning_rate,
        }
        # convert string into a list
        encoder_lr_schedules = args.encoder_lr_schedules.split()
        encoder_lr_schedules_list = [int(e) for e in encoder_lr_schedules]
        decoder_lr_schedules = args.decoder_lr_schedules.split()
        decoder_lr_schedules_list = [int(e) for e in decoder_lr_schedules]
        self.lr_schedules = {
            "encoder": encoder_lr_schedules_list,
            "decoder": decoder_lr_schedules_list,
            "text_span_decoder": decoder_lr_schedules_list,
        }
        self.warmup_steps = {
            "encoder": args.encoder_warmup_steps,
            "decoder": args.decoder_warmup_steps,
            "text_span_decoder": args.decoder_warmup_steps,
        }
        if args.optimizer == "adam":
            self.optimizers = {
                "encoder": Adam(model.encoder.parameters(), lr=self.lr["encoder"]),
                "decoder": Adam(model.decoder.parameters(), lr=self.lr["decoder"]),
                "text_span_decoder": Adam(
                    model.decoder.parameters(), lr=self.lr["text_span_decoder"]
                ),
            }
        elif args.optimizer == "adagrad":
            self.optimizers = {
                "encoder": Adagrad(model.encoder.parameters(), lr=self.lr["encoder"]),
                "decoder": Adagrad(model.decoder.parameters(), lr=self.lr["decoder"]),
                "text_span_decoder": Adam(
                    model.decoder.parameters(), lr=self.lr["text_span_decoder"]
                ),
            }
        else:
            raise NotImplementedError

        self._step = 0

    def _update_rate(self, stack):
        if self._step < self.warmup_steps[stack]:
            return self.lr[stack] * self._step / self.warmup_steps[stack]
        else:
            factor = 1
            for schedule in self.lr_schedules[stack]:
                if self._step > schedule:
                    factor /= 10.0
                    break
            return self.lr[stack] * factor

    def zero_grad(self):
        self.optimizer_decoder.zero_grad()
        self.optimizer_encoder.zero_grad()

    def step(self):
        self._step += 1
        for stack, optimizer in self.optimizers.items():
            new_rate = self._update_rate(stack)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_rate
            optimizer.step()
