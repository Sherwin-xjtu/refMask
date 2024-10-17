bedtools intersect -a new_file.txt -b NA12878.recal.merged.new.sorted.chrX.bam -c -sorted > counts.txt
bedtools intersect -a new_file.txt -b NA12878.recal.merged.new.sorted.q0.b.chrX.bam -c -sorted > q0counts.txt
