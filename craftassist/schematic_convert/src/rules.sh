#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.


> all_swaps.txt

_permute() {
    for i in $*
    do
        for j in $*
        do
            echo $i $j | sed -e "s/_/ /g" >> all_swaps.txt
        done
    done
}

lala=$(grep "stonebrick\|stone\[\|cobblestone$\|double_stone_slab\[\|clay\[\|clay$$\|brick_block$" idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala

lala=$(grep "obsidian\|nether_brick$\|marine\|purpur_block\|_pillar" idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lolo

#weird block can be replaced by standard ones:
for i in $lolo
do
    for j in $lala
    do
        echo $i $j | sed -e "s/_/ /g" >> all_swaps.txt
    done
done

#stairs
for axis in north east west south
do
    for half in top  bottom
    do
        lala=$(grep "stair" idx_blocks.txt | grep "facing=${axis},half=${half}" | awk '{print $1"_-1"}')
        _permute $lala
    done
done


#gravel / sand
lala=$(grep "^13 \|^12 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala

# ore
lala=$(grep "_ore" idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala

#block
#beacon blocks:
lala=$(grep "_block"  idx_blocks.txt | grep "iron\|gold\|emerald\|diamond" | awk '{print $1"_"$2}')
_permute $lala
#other blocks:
lolo=$(grep "_block"  idx_blocks.txt | grep -v "iron\|gold\|emerald\|diamond\|command\|structure\|mushroom\|hay\|melon\|redstone" | awk '{print $1"_"$2}')
_permute $lolo

#other can be replaced by special beacon blocks
for i in $lolo
do
    for j in $lala
    do
        echo $i $j | sed -e "s/_/ /g" >> all_swaps.txt
    done
done

#mushroom block
lala=$(grep "mushroom" idx_blocks.txt | grep "block"  | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "mushroom" idx_blocks.txt | grep -v "block"  | awk '{print $1"_"$2}')
_permute $lala

lala=$(grep "plant" idx_blocks.txt | grep "red\|yellow" | awk '{print $1"_"$2}')
_permute $lala

# log and planks
lala=$(grep "planks\|double_wooden_slab" idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala

for axis in x y z none
do
    lala=$(grep "log" idx_blocks.txt | grep "axis=${axis}" | awk '{print $1"_"$2}')
    _permute $lala
done


#glass
lala=$(grep "glass" idx_blocks.txt | grep -v "pane" | awk '{print $1"_"$2}')
_permute $lala

lala=$(grep "glass" idx_blocks.txt | grep "pane" | awk '{print $1"_"$2}')
_permute $lala


#plants
lala=$(grep "^81 \|^83 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala

# swap within:
#wool
lala=$(grep "wool" idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^3 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^6 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^18 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^43 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^31 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^59 \|^60 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^78 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^92 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^97 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^105 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^127 \|^141 \|^142 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^115 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^117 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^118 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^125 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^127 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^141 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^142 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^161 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^170 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^171 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^175 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^200 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^207 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala
lala=$(grep "^212 " idx_blocks.txt | awk '{print $1"_"$2}')
_permute $lala

for axis in bottom top
do
    lala=$(grep "slab" idx_blocks.txt | grep -v "double" | awk '{print $1"_"$2}')
    _permute $lala
done

