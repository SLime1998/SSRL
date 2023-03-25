date

#### AimCLR NTU-60 xsub ####

# Pretext
#python main.py pretrain_byolclr_tri --config config/ntu60/pretext/pretext_aim_xsub_motion.yaml
#python main.py pretrain_byolclr_tri --config config/ntu60/pretext/pretext_aim_xview_motion.yaml
#python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_aimclr_xview_joint.yaml

#python main.py pretrain_byolclr_tri --config config/ntu60/pretext/pretext_aim_xsub_bone.yaml
#python main.py pretrain_byolclr_tri --config config/ntu60/pretext/pretext_aim_xview_bone.yaml
#python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_aimclr_xsub_bone.yaml
#python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_aimclr_xview_bone.yaml

#python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_aimclr_xsub_motion.yaml
#python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_aimclr_xview_motion.yaml

#python main.py linear_evaluation --config config/ntu60/linear_eval/semi_sticlr_xsub_joint.yaml
#python main.py linear_evaluation --config config/ntu60/linear_eval/semi_sticlr_xsub_bone.yaml
#python main.py linear_evaluation --config config/ntu60/linear_eval/semi_sticlr_xsub_motion.yaml

#python main.py pretrain_byolclr_tri --config config/ntu60/pretext/pretext_sticlr_xsub_bone.yaml
#python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_sticlr_xsub_bone.yaml


#python main.py pretrain_byolclr_tri --config config/ntu60/pretext/pretext_sticlr_xsub_joint.yaml
#python main.py pretrain_byolclr_tri --config config/ntu60/pretext/pretext_sticlr_xsub_motion.yaml
#python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_sticlr_xsub_motion.yaml

#python main.py pretrain_byolclr_tri --config config/ntu60/pretext/second01.yaml


# Linear_eval
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_sticlr_xsub_joint.yaml
#python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_sticlr_xsub_motion.yaml
#python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_sticlr_xsub_bone.yaml

# Ensemble
#python ensemble_ntu_cs.py
