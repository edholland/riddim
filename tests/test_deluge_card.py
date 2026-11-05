from riddim.deluge.card import pack7, unpack7


def test_card_bytes_survive_the_seven_bit_packing():
    data = bytes(range(256)) + b"\xff\x00\x80"
    packed = pack7(data)
    assert all(b < 128 for b in packed)
    assert unpack7(bytes(packed)) == data
