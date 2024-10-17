bedtools bamtobed -i NA12878.recal.merged.new.t2t.sorted.q0.b.bam > NA12878.recal.merged.new.t2t.sorted.q0.b.bed
bgzip -c  NA12878.recal.merged.new.t2t.sorted.q0.b.bed > NA12878.recal.merged.new.t2t.sorted.q0.b.bed.gz
tabix -p bed NA12878.recal.merged.new.t2t.sorted.q0.b.bed.gz
bedtools nuc -fi /czlab/xuwen/references/chm13/chm13v2.fa -bed NA12878.recal.merged.new.t2t.sorted.q0.b.bed.gz > NA12878.recal.merged.new.t2t.sorted.q0.b.bed.repeats.txt
