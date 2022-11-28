from os import path, environ
import pandas as pd

datasets = ['cifarfs', 'fc100', 'miniimagenet']# 'miniimagenet84',]# 'tieredimagenet']
batch_size = 125

win_df = pd.read_csv('results/win.csv')
print(win_df.to_markdown())
with open('scripts/run.bash', 'w') as fp:
    for i in [1, ]:
        for D in datasets:
            for M in ['']:
                for N in [0.0,]:
                    for NT in [N]:
                        for X in [0]:
                            for S in ['00.0']:
                                for R in [0.0,]:
                                    for A in [0]:
                                        B = A
                                        for G in [0]:
                                            for E in [1]:
                                                for V in [0, ]:
                                                    for P in [0]:
                                                        for T in [1]:
                                                            ddf = win_df[(win_df.N == N) & (win_df.D == D)]
                                                            if ddf.A.item() == 1:
                                                                model_fname = f'model.{D}_{ddf.R.item()}_n{N}_x{ddf.X.item()}_s00.0_a{ddf.A.item()}_b{ddf.B.item()}_g{ddf.G.item()}_#1.pt55'
                                                            else:
                                                                model_fname = f'model.{D}_{ddf.R.item()}_n{N}_x{ddf.X.item()}_s00.0_#1.pt55'
                                                                features_fname = f'features.{D}_{ddf.R.item()}_n{N}_x{ddf.X.item()}_s00.0_#1.pt55'
                                                            model_pname = path.join(environ['HOME'],'checkpoints',model_fname)
                                                            features_pname = path.join(environ['HOME'],'checkpoints',features_fname)
                                                            cmd = f'D={D:<16}       ; X={X}; R={R}; S={S}; M={M}    ; N={N}; NT={NT}; A={A}; B={B}; G={G:<5}; E={E}; V={V}; P={P}; T={T}; C=3;'\
                                                                f'NAME=${{D}}${{M}}_${{R}}_n${{N}}_x${{X}}_s${{S}}_a${{A}}_b${{B}}_g${{G}}_e${{E}}_v${{V}}_p${{P}}_t${{T}}_#{i}'\
                                                                f' && python main.py --dataset-path ~/data/easy'\
                                                                f' {f"--transformer --test-features {features_pname} --batch-fs 200 --dataset-device cpu" if T else ""}'\
                                                                f' {f"--load-model {model_pname}" if P else ""}'\
                                                                f' --dataset ${{D}} --label-noise ${{N}} ${{M}}'\
                                                                f' --model resnet12 {"--episodic" if E else ""}'\
                                                                f' {"--protonet-no-square" if V else ""} --symmetric-loss ${{S}}'\
                                                                f' --apl-alpha ${{A}}'\
                                                                f' --apl-beta ${{B}}'\
                                                                f' --star-loss-gamma ${{G}}'\
                                                                f'  --rotations ${{R}}'\
                                                                f' --epochs 100  --manifold-mixup ${{X}}'\
                                                                f' --cosine --gamma 0.9 --milestones 100 --batch-size {batch_size} --preprocessing EM --n-shots 5 --skip-epochs 98  --save-model ${{HOME}}'\
                                                                f'/checkpoints/model.${{NAME}}'\
                                                                f'.pt5 --save-features ${{HOME}}'\
                                                                f'/checkpoints/features.${{NAME}}'\
                                                                f'.pt5 --n-unknown 0 --device cuda:${{C}}'\
                                                                f' --lr 0.01  --n-runs 1000 > logs/nohup.${{NAME}}'\
                                                                f'.out && python main.py --dataset-path  ~/data/easy --dataset ${{D}}'\
                                                                f'  --model resnet12 --test-features /home/ubuntu/checkpoints/features.${{NAME}}'\
                                                                f'.pt55 --preprocessing ME --n-shots "[5,]" --label-noise-test ${{NT}}'\
                                                                f' --device cuda:${{C}}'\
                                                                f' --n-unknown 0 --lam 1000 --svm-c -1  --n-runs 1000'                                                  
                                                        print(cmd, file=fp)
                                                    