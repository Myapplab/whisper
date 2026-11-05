# whisper multilingual.tiktoken -> HF tokenizers tokenizer.json (faster-whisper icin)
import sys, json
import importlib.util, glob, os
_wt = glob.glob(os.path.join(sys.prefix, "lib", "python3.*", "site-packages", "whisper", "tokenizer.py"))[0]
_spec = importlib.util.spec_from_file_location("wtok", _wt); _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
get_encoding = _m.get_encoding
from transformers.convert_slow_tokenizer import TikTokenConverter
from tokenizers import AddedToken
enc = get_encoding("multilingual")
path = os.path.join(os.path.dirname(_wt), "assets", "multilingual.tiktoken")
pat = enc._pat_str
conv = TikTokenConverter(vocab_file=path, pattern=pat, additional_special_tokens=[])
tok = conv.converted()
specials = sorted(enc._special_tokens.items(), key=lambda kv: kv[1])
base = tok.get_vocab_size()
print("base", base, "first special", specials[0])
assert specials[0][1] == base, (specials[0], base)
tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for t, _ in specials])
for t, i in specials[:3] + specials[-2:]:
    assert tok.token_to_id(t) == i, (t, i, tok.token_to_id(t))
s = "Hello world, ask not what your country can do. Merhaba dünya, nasılsınız?"
assert tok.encode(s, add_special_tokens=False).ids == enc.encode(s), "encode mismatch"
assert tok.decode(enc.encode(s)) == s
tok.save(sys.argv[1])
print("ok", tok.get_vocab_size())
