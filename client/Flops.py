
class Flops:
    def __init__(self, count):
        self.count = count
        self.flops_forward_epoch, self.flops_encoder_epoch, self.flops_backprop_epoch, self.flops_rest, self.flops_send, self.flops_recieve = 0,0,0,0,0,0
        if count:
            # Imports to count FLOPs
            # Does not work on every architecture
            # The paranoid switch prevents the FLOPs count
            # Solution: sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
            from ptflops import get_model_complexity_info
            from pypapi import events, papi_high as high
            # Starts internal FLOPs counter | If there is an Error: See "from pypapi import events"
            self.high.start_counters([events.PAPI_FP_OPS,])

    def read_counter(self, context):
        if self.count:
            x = self.high.read_counters()
            if context == "forward":
                self.flops_forward_epoch += x[0]
            if context == "rest":
                self.flops_rest += x[0]
            if context == "encoder":
                self.flops_encoder_epoch += x[0]
            if context == "send":
                self.flops_send += x[0]
            if context == "recieve":
                self.flops_recieve += x[0]
            if context == "backprop":
                self.flops_backprop_epoch += x[0]

    def reset(self):
        self.flops_forward_epoch, self.flops_encoder_epoch, self.flops_backprop_epoch, self.flops_rest, self.flops_send, self.flops_recieve = 0,0,0,0,0,0




