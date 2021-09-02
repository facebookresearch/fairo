source activate ~/.conda/envs/locobot_env

export EXPPREFIX=_mul_traj

# export OUTDIRx1=/checkpoint/apratik/finals/straightline/apartment_0/pred_label_gt5p2fix$EXPPREFIX
# python label_propagation.py --scene_path $SCENE_ROOT --gtframes 5 --propogation_step 2 --out_dir $OUTDIRx1

# export OUTDIRx2=/checkpoint/apratik/finals/straightline/apartment_0/pred_label_gt10p2fix$EXPPREFIX
# python label_propagation.py --scene_path $SCENE_ROOT --gtframes 10 --propogation_step 2 --out_dir $OUTDIRx2

# export OUTDIRx3=/checkpoint/apratik/finals/straightline/apartment_0/pred_label_gt15p2fix$EXPPREFIX
# python label_propagation.py --scene_path $SCENE_ROOT --gtframes 15 --propogation_step 2 --out_dir $OUTDIRx3

# export OUTDIRx4=/checkpoint/apratik/finals/straightline/apartment_0/pred_label_gt20p2fix$EXPPREFIX
# python label_propagation.py --scene_path $SCENE_ROOT --gtframes 20 --propogation_step 2 --out_dir $OUTDIRx4

# export OUTDIRx5=/checkpoint/apratik/finals/straightline/apartment_0/pred_label_gt25p2fix$EXPPREFIX
# python label_propagation.py --scene_path $SCENE_ROOT --gtframes 25 --propogation_step 2 --out_dir $OUTDIRx5


export SCENE_ROOTD=/checkpoint/apratik/data/apartment_0/default/no_noise_test_mul_traj
# echo $SCENE_ROOTD 

END=19
for i in $(seq $END); 
    do echo $SCENE_ROOTD/$i
    for gt in 5 10 15 20 25
        do
        export OUTDIR=$SCENE_ROOTD/$i/pred_label_gt${gt}p2fix$EXPPREFIX
        echo $OUTDIR
        python label_propagation.py --scene_path $SCENE_ROOTD/$i --gtframes $gt --propogation_step 2 --out_dir $OUTDIR
    done    
done


# export OUTDIRy1=/checkpoint/apratik/finals/default/apartment_0/pred_label_gt5p2fix$EXPPREFIX
# # python label_propagation.py --scene_path $SCENE_ROOTD --gtframes 5 --propogation_step 2 --out_dir $OUTDIRy1

# export OUTDIRy2=/checkpoint/apratik/finals/default/apartment_0/pred_label_gt10p2fix$EXPPREFIX
# python label_propagation.py --scene_path $SCENE_ROOTD --gtframes 10 --propogation_step 2 --out_dir $OUTDIRy2

# export OUTDIRy3=/checkpoint/apratik/finals/default/apartment_0/pred_label_gt15p2fix$EXPPREFIX
# python label_propagation.py --scene_path $SCENE_ROOTD --gtframes 15 --propogation_step 2 --out_dir $OUTDIRy3

# export OUTDIRy4=/checkpoint/apratik/finals/default/apartment_0/pred_label_gt20p2fix$EXPPREFIX
# python label_propagation.py --scene_path $SCENE_ROOTD --gtframes 20 --propogation_step 2 --out_dir $OUTDIRy4

# export OUTDIRy5=/checkpoint/apratik/finals/default/apartment_0/pred_label_gt25p2fix$EXPPREFIX
# python label_propagation.py --scene_path $SCENE_ROOTD --gtframes 25 --propogation_step 2 --out_dir $OUTDIRy5