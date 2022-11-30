from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetTransporterAgent
from nat_policies.models.roboclip import TwoStreamAttentionLangFusionLat
from nat_policies.models.roboclip import TwoStreamTransportLangFusionLat


class RoboclipAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'roboclip_lingunet_lat'
        self.attention = TwoStreamAttentionLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )