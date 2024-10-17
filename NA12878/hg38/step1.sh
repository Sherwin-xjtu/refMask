bedtools bamtobed -i NA12878.recal.merged.new.sorted.q0.b.chrX.bam > NA12878.recal.merged.new.sorted.q0.b.chrX.q0.b.bed
bgzip -c  NA12878.recal.merged.new.sorted.q0.b.chrX.q0.b.bed > NA12878.recal.merged.new.sorted.q0.b.chrX.q0.b.bed.gz
tabix -p NA12878.recal.merged.new.sorted.q0.b.chrX.q0.b.bed.gz
/data/programs/bedtools2/2.29.0/bin/bedtools merge -i NA12878.recal.merged.new.sorted.q0.b.chrX.q0.b.bed.gz -d 1000 > NA12878.recal.merged.new.sorted.q0.b.chrX.q0.b.merged.bed
/data/programs/bedtools2/2.29.0/bin/bedtools nuc -fi /czlab/xuwen/references/hg38/v0/Homo_sapiens_assembly38.fasta -bed NA12878.recal.merged.new.sorted.q0.b.chrX.q0.b.merge.bed > NA12878.recal.merged.new.sorted.q0.b.chrX.q0.b.merged.bed.repeats.txt
bedtools map -a NA12878.recal.merged.new.sorted.q0.b.chrX.q0.b.merged.bed  -b NA12878.recal.merged.new.sorted.chrX.bam -c 5 -o mean > output_fNA12878.recal.merged.new.sorted.chrX.intervals.mean.txt
