export SCENE_ROOT=/checkpoint/apratik/finals/straightline
export EXPPREFIX=_corl
echo $SCENE_ROOT 

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


export SCENE_ROOTD=/checkpoint/apratik/finals/default
echo $SCENE_ROOTD 

export OUTDIRy1=/checkpoint/apratik/finals/default/apartment_0/pred_label_gt5p2fix$EXPPREFIX
python label_propagation.py --scene_path $SCENE_ROOTD --gtframes 5 --propogation_step 2 --out_dir $OUTDIRy1

export OUTDIRy2=/checkpoint/apratik/finals/default/apartment_0/pred_label_gt10p2fix$EXPPREFIX
python label_propagation.py --scene_path $SCENE_ROOTD --gtframes 10 --propogation_step 2 --out_dir $OUTDIRy2

export OUTDIRy3=/checkpoint/apratik/finals/default/apartment_0/pred_label_gt15p2fix$EXPPREFIX
python label_propagation.py --scene_path $SCENE_ROOTD --gtframes 15 --propogation_step 2 --out_dir $OUTDIRy3

export OUTDIRy4=/checkpoint/apratik/finals/default/apartment_0/pred_label_gt20p2fix$EXPPREFIX
python label_propagation.py --scene_path $SCENE_ROOTD --gtframes 20 --propogation_step 2 --out_dir $OUTDIRy4

export OUTDIRy5=/checkpoint/apratik/finals/default/apartment_0/pred_label_gt25p2fix$EXPPREFIX
python label_propagation.py --scene_path $SCENE_ROOTD --gtframes 25 --propogation_step 2 --out_dir $OUTDIRy5