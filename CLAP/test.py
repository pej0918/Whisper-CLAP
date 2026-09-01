import laion_clap

model = laion_clap.CLAP_Module(enable_fusion=True, amodel= 'HTSAT-tiny') #enable_fusion=False
model.load_ckpt('/data1/eunju/model_ckpt/CLAP/630k-fusion-best.pt') # 630k-audioset-fusion-best.pt 630k-audioset-best.pt
