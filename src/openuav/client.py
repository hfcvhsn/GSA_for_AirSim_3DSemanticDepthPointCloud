"""AirSim 1.8.1 API with the modern-msgpack RPC fork, without site-package edits."""
import airsim
import msgpackrpc


class OpenUAVClient(airsim.MultirotorClient):
    def __init__(self, ip="127.0.0.1", port=41451, timeout_value=30):
        # Official AirSim passes pack_encoding/unpack_encoding removed by the fixed
        # RPC fork. VehicleClient.__init__ otherwise only creates self.client.
        self.client = msgpackrpc.Client(msgpackrpc.Address(ip or "127.0.0.1", port), timeout=timeout_value)
