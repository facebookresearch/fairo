export EXPPREFIX=_h1

echo $SCENE_ROOT 

export FRAMES=5
for scene in straightline default
do 
    for p in 2 4 6 8
    do 
        export OUTDIR=/checkpoint/apratik/finals/$scene/apartment_0/pred_label_gt5p$p$EXPPREFIX
        echo $OUTDIR
        python label_propagation.py --scene_path /checkpoint/apratik/finals/$scene --gtframes $FRAMES --propogation_step $p --out_dir $OUTDIR
    done
done