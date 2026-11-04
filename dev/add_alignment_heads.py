import base64, gzip, json, re, sys, glob, os, numpy as np
src = open(glob.glob(os.path.join(sys.prefix, "lib/python3*/site-packages/whisper/__init__.py"))[0]).read()
blob = re.search(r'"small":\s*b"([^"]+)"', src).group(1).encode()
arr = np.frombuffer(gzip.decompress(base64.b85decode(blob)), dtype=bool).reshape(12, 12)
heads = [[int(l), int(h)] for l, h in zip(*np.nonzero(arr))]
cfgp = sys.argv[1]; cfg = json.load(open(cfgp)); cfg["alignment_heads"] = heads; json.dump(cfg, open(cfgp, "w"))
print(heads)
