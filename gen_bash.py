datasets = ['cifarfs', 'fc100', 'miniimagenet']# 'miniimagenet84',]# 'tieredimagenet']
with open('scripts/run.bash', 'w') as fp:
    for i in [1, 2, 3, ]:
        for D in datasets:
            for M in ['']:
                for N in [0.0, 0.2, 0.4, 0.6]:
                    for NT in [N]:
                        for X in [0]:
                            for S in ['00.0']:
                                for R in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                                    for A in [1]:
                                        for B in [1]:
                                            for G in [0]:
                                                cmd = f'D={D:<16}       ; X={X}; R={R}; S={S}; M={M}    ; N={N}; NT={NT}; A={A}; B={B}; G={G:<5}; C=3;  NAME=${{D}}${{M}}_${{R}}_n${{N}}_x${{X}}_s${{S}}_a${{A}}_b${{B}}_g${{G}}_#{i}; python main.py --dataset-path ~/data/easy --dataset ${{D}} --label-noise ${{N}} ${{M}} --model resnet12 --symmetric-loss ${{S}} --apl-alpha ${{A}} --apl-beta ${{B}} --star-loss-gamma ${{G}}  --rotations ${{R}} --epochs 100  --manifold-mixup ${{X}} --cosine --gamma 0.9 --milestones 100 --batch-size 128 --preprocessing EM --n-shots 5 --skip-epochs 98  --save-model ${{HOME}}/checkpoints/model.${{NAME}}.pt5 --save-features ${{HOME}}/checkpoints/features.${{NAME}}.pt5 --n-unknown 0 --device cuda:${{C}} --lr 0.1  --n-run 1000 > logs/nohup.${{NAME}}.out; python main.py --dataset-path  ~/data/easy --dataset ${{D}}  --model resnet12 --test-features /home/ubuntu/checkpoints/features.${{NAME}}.pt55 --preprocessing ME --n-shots "[5,]" --label-noise-test ${{NT}} --device cuda:${{C}} --n-unknown 0 --lam 1000 --svm-c -1  --n-runs 1000'
                                                print(cmd, file=fp)
                                                    