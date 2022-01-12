Overall Pipeline 

1. Data collection - one time
    \baseline
        \0
        \1
        
2. Process baseline to find spawn, target locations (based on candidate heuristic, can be multiple) 
    \job_folder
        \baseline 
            \0
                \rgb
                ...
                \class
                    \GT5
                        \reexplore5.json
                    \GT10
                        \reexplore10.json

                \instance
                    \GT5
                        \reexplore5.json
                    \GT10
                    ...
            \1
        
3. Do Reexplore
    \job_folder
        \baseline
            \0
                \class
                    \5
                        \S1
                        \C1
                        \..
                        \reexplore5.json
                    \10
                        \reexplore10.json

                \instance
                    \5
                        \S1
                        \C1
                        ..
                        \reexplore5.json
                    ...
            \1
        
    Bugs:
    * why is there a < 100% success rate? 


4. Label Prop - create all gtp combinations 
    propagate each folder and create a predlabel folder
    then copy over combinations using the predlabel folders
    
   \job_folder
        \baseline
            \0
                \class
                    \5
                        \0
                            \S1
                                \rgb
                                \seg
                                \predlabelgt5px
                            \C1
                            \..
                        \1
                            ..
                        \reexplore5.json
                    \10
                        \reexplore10.json


                \instance
                    \5
                        \S1
                        \C1
                        ..
                        \reexplore5.json
                    ...
            \1
  
5. Combine all predlabels into one
   \job_folder
        \baseline
            \0
                \class
                    \5
                        \a
                            \gt5p2
                            \gt5p4
                            ...
                        \b 
                            \gt5p2
                            ...
                        \S1
                        \C1
                        \..
                        \reexplore5.json
                    \10
                        \reexplore10.json


                \instance
                    \5
                        \S1
                        \C1
                        ..
                        \reexplore5.json
                    ...
            \1

            
6. coco-ize + train
    \job_folder
        \baseline
            \0
                \class
                    \5
                        \a
                            \gt5p2
                                \rgb
                                \seg
                                coco_train.json
                                metrics.json
                            \gt5p4
                            ...
                        \b 
                            \gt5p2
                            ...
                        \S1
                        \C1
                        \..
                        \reexplore5.json
                    \10
                        \reexplore10.json


                \instance
                    \5
                        \S1
                        \C1
                        ..
                        \reexplore5.json
                    ...
            \1

            
7. Visualization
    outcome - compare e1+r1+r2 vs e1+s1+r2 vs e1+c1+r2 vs e1+c1+s1
            

1. Find candidates and respawn locations
```
./launch_candidates.sh /checkpoint/apratik/data_reexplore/baselinev3
```

2. Launch Reexplore
```
./launch_reexplore.sh <outdir from step 1> <noise or not>
./launch_reexplore.sh /checkpoint/apratik/jobs/reexplore/respawnv3/baselinev3
```

3. Label Prop