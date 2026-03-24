import unittest

from sglang.multimodal_gen.runtime.disaggregation.dispatch_policy import (
    MaxFreeSlotsFirst,
    PoolDispatcher,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import (
    RoleType,
    filter_modules_for_role,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferPeerInfoMsg,
    TransferRegisterMsg,
    decode_transfer_msg,
    encode_transfer_msg,
    is_transfer_message,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class TestDisaggControlPlane(unittest.TestCase):
    def test_transfer_register_round_trip_preserves_direct_connect_fields(self):
        msg = TransferRegisterMsg(
            role="encoder",
            instance_id=7,
            session_id="sess-1",
            pool_ptr=1234,
            pool_size=4096,
            control_endpoint="tcp://10.0.0.1:9001",
            work_endpoint="tcp://10.0.0.1:9000",
            rank0_only=True,
            role_device="cpu",
            preallocated_slots=[
                {
                    "offset": 512,
                    "size": 2048,
                    "slot_id": 3,
                    "addr": 1746,
                }
            ],
        )

        frames = encode_transfer_msg(msg)
        decoded = decode_transfer_msg(frames)

        self.assertTrue(is_transfer_message(frames))
        self.assertEqual(decoded["role"], "encoder")
        self.assertEqual(decoded["instance_id"], 7)
        self.assertEqual(decoded["control_endpoint"], "tcp://10.0.0.1:9001")
        self.assertEqual(decoded["work_endpoint"], "tcp://10.0.0.1:9000")
        self.assertTrue(decoded["rank0_only"])
        self.assertEqual(decoded["role_device"], "cpu")
        self.assertEqual(decoded["preallocated_slots"][0]["slot_id"], 3)

    def test_transfer_peer_info_round_trip_preserves_receiver_fields(self):
        msg = TransferPeerInfoMsg(
            request_id="req-1",
            dest_session_id="dst-session",
            dest_addr=987654,
            transfer_size=8192,
            receiver_role="decoder",
            receiver_instance=2,
            receiver_control_endpoint="tcp://10.0.0.3:9101",
            prealloc_slot_id=5,
        )

        decoded = decode_transfer_msg(encode_transfer_msg(msg))

        self.assertEqual(decoded["request_id"], "req-1")
        self.assertEqual(decoded["transfer_size"], 8192)
        self.assertEqual(decoded["receiver_role"], "decoder")
        self.assertEqual(decoded["receiver_instance"], 2)
        self.assertEqual(
            decoded["receiver_control_endpoint"],
            "tcp://10.0.0.3:9101",
        )
        self.assertEqual(decoded["prealloc_slot_id"], 5)

    def test_server_args_resolve_control_endpoint_and_role_device(self):
        args = object.__new__(ServerArgs)
        args.scheduler_port = 31000
        args.host = "0.0.0.0"
        args.disagg_p2p_hostname = "10.1.2.3"
        args.num_gpus = 0
        args.disagg_role_device = "auto"

        self.assertEqual(
            args.derive_pool_control_endpoint(),
            "tcp://0.0.0.0:31001",
        )
        self.assertEqual(
            args.derive_pool_control_advertised_endpoint(),
            "tcp://10.1.2.3:31001",
        )
        self.assertEqual(args.resolved_role_device(), "cpu")

        args.num_gpus = 4
        self.assertEqual(args.resolved_role_device(), "cuda")

        args.disagg_role_device = "cpu"
        self.assertEqual(args.resolved_role_device(), "cpu")

    def test_max_free_slots_policy_uses_explicit_capacity(self):
        policy = MaxFreeSlotsFirst(num_instances=3, max_slots_per_instance=4)
        self.assertEqual(policy.select(active_counts=[4, 1, 2]), 1)
        self.assertIsNone(policy.select_with_capacity([0, 0, 0]))

        dispatcher = PoolDispatcher(
            num_encoders=1,
            num_denoisers=3,
            num_decoders=1,
            policy_name="max_free_slots",
            max_slots_per_instance=4,
        )
        self.assertEqual(dispatcher.select_denoiser(active_counts=[4, 1, 2]), 1)

    def test_encoder_module_filtering_requires_decoder_opt_in(self):
        module_names = ["text_encoder", "vae", "transformer", "scheduler"]

        filtered_default = filter_modules_for_role(module_names, RoleType.ENCODER)
        filtered_with_decoder = filter_modules_for_role(
            module_names,
            RoleType.ENCODER,
            extra_allowed_modules={"vae"},
        )

        self.assertEqual(filtered_default, ["text_encoder", "scheduler"])
        self.assertEqual(
            filtered_with_decoder,
            ["text_encoder", "vae", "scheduler"],
        )


if __name__ == "__main__":
    unittest.main()
