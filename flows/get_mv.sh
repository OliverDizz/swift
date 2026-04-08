# indir=/home/cywu/cywu/video_compression/trace_yuv/cif/all_png_frames/
# outdir=/home/cywu/cywu/video_compression/trace_yuv/cif/all_motion_vector_images/
# outdir=/home/cywu/cywu/video_compression/trace_yuv/cif/all_motion_vector_images2/
# tmp_prefix='trace'
# mode=8digits

indir=/nfs/home/olidiz18/git/swift/codec/src/data/eval
outdir=/nfs/home/olidiz18/git/swift/codec/src/data/eval_mv
tmp_prefix=frame$1_
mode=4digits
type=png

mkdir -p ${outdir}

echo $1

for f in ${indir}/$1*.${type}
do
    video=${f##*/}
    if [ $mode = "8digits" ]; then
        video_name=${video::-12}
    fi

    if [ $mode = "4digits" ]; then
        video_name=${video::-8}
    fi

    name=${video::-4}
    idx=${name##*_}
    idx=$(( 10#$idx ))

    echo 'video ' $video
    echo 'video_name ' $video_name
    echo 'name ' $name
    echo 'idx ' $idx
    group_idx=$(($idx%12))
    
    if [ $group_idx = "1" ]; then
        echo 'idx group 1 ' $idx

        # 0 is I-frame1, 12 is I-frame2.
        last_idx=$(($idx+12))

        if [ $mode = "8digits" ]; then
            last_frame=${indir}/${video_name}$(printf "%08d" ${last_idx}).${type}
        fi
        if [ $mode = "4digits" ]; then
            last_frame=${indir}/${video_name}$(printf "%04d" ${last_idx}).${type}
        fi

        echo 'last frame ' ${last_frame}
        if [ -f ${last_frame} ]; then
            for i in `seq 1 11`; do
                cur_idx=$(($idx+$i))
                if [ $i = "1" ] || [ $i = "4" ] || [ $i = "7" ] || [ $i = "10" ]; then
                    prev_idx=$(($cur_idx-1))
                    next_idx=$(($cur_idx+2))
                fi
                if [ $i = "2" ] || [ $i = "5" ] || [ $i = "8" ] || [ $i = "11" ]; then
                    prev_idx=$(($cur_idx-2))
                    next_idx=$(($cur_idx+1))
                fi
                if [ $i = "3" ] || [ $i = "9" ]; then
                    prev_idx=$(($cur_idx-3))
                    next_idx=$(($cur_idx+3))
                fi
                if [ $i = "6" ]; then
                    prev_idx=$(($cur_idx-6))
                    next_idx=$(($cur_idx+6))
                fi
                
                if [ $mode = "8digits" ]; then
                    cur_frame=${indir}/${video_name}$(printf "%08d" ${cur_idx}).${type}
                    prev_frame=${indir}/${video_name}$(printf "%08d" ${prev_idx}).${type}
                    next_frame=${indir}/${video_name}$(printf "%08d" ${next_idx}).${type}
                fi
                if [ $mode = "4digits" ]; then
                    cur_frame=${indir}/${video_name}$(printf "%04d" ${cur_idx}).${type}
                    prev_frame=${indir}/${video_name}$(printf "%04d" ${prev_idx}).${type}
                    next_frame=${indir}/${video_name}$(printf "%04d" ${next_idx}).${type}
                fi
                
                cp $prev_frame tmp_${tmp_prefix}1.${type}
                cp $cur_frame tmp_${tmp_prefix}2.${type}
                rm -f tmp_${tmp_prefix}3.${type}  # Added -f to prevent errors

                cur_frame_name=${cur_frame##*/}
                cur_frame_name=${cur_frame_name::-4}
                echo 'extracting before_flow for ' $cur_frame_name

                # --- BEFORE FLOW EXTRACTION ---
                yes | ffmpeg -hide_banner -loglevel error -i "tmp_${tmp_prefix}%01d.${type}" -c:v libx264 -g 2 -bf 0 -b_strategy 0 -sc_threshold 0 tmp_${tmp_prefix}.mp4
                
                /nfs/home/olidiz18/git/swift/flows/MV_extract/ffmpeg-2.7.2/extract_mvs tmp_${tmp_prefix}.mp4 >tmp_${tmp_prefix}.mvs0
                
                /nfs/home/olidiz18/git/swift/flows/MV_extract/MV-code-release/MotionVector \
                            tmp_${tmp_prefix}.mvs0 ${outdir}/${cur_frame_name}_before_flow_x ${outdir}/${cur_frame_name}_before_flow_y


                echo 'extracting after_flow for ' $cur_frame_name
                
                cp $next_frame tmp_${tmp_prefix}1.${type}
                
                yes | ffmpeg -hide_banner -loglevel error -i "tmp_${tmp_prefix}%01d.${type}" -c:v libx264 -g 2 -bf 0 -b_strategy 0 -sc_threshold 0 tmp_${tmp_prefix}.mp4
                
                /nfs/home/olidiz18/git/swift/flows/MV_extract/ffmpeg-2.7.2/extract_mvs tmp_${tmp_prefix}.mp4 >tmp_${tmp_prefix}.mvs0
                
                /nfs/home/olidiz18/git/swift/flows/MV_extract/MV-code-release/MotionVector \
                            tmp_${tmp_prefix}.mvs0 ${outdir}/${cur_frame_name}_after_flow_x ${outdir}/${cur_frame_name}_after_flow_y

            done
        fi
    fi
done

# Clean up temp files when done
rm -f tmp_${tmp_prefix}*