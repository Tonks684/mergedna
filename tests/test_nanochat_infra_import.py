def test_nanochat_flash_attn_imports():
    from nanochat_infra.flash_attention import flash_attn_func
    assert flash_attn_func is not None