gatk_path=/homes8/xuwen/software/gatk/gatk
input_mapped_merged_marked_bam=/czlab/xuwen/results/NA12878/300X/outDir/NA12878/bams/NA12878.recal.merged.new.t2t.sorted.bam
ref_fasta=/czlab/xuwen/references/chm13/chm13v2.fa

srun --cpus-per-task=8 --mem-per-cpu=8G \
${gatk_path} --java-options "-Dsamjdk.compression_level=5 -Xms50000m -Djava.io.tmpdir=/czlab/xuwen/tmp" \
    SortSam \
--INPUT ${input_mapped_merged_marked_bam} \
--OUTPUT /dev/stdout \
--SORT_ORDER "coordinate" \
--CREATE_INDEX false \
--CREATE_MD5_FILE false \
| \
${gatk_path} --java-options "-Dsamjdk.compression_level=5 -Xms50000m -Djava.io.tmpdir=/czlab/xuwen/tmp" \
SetNmMdAndUqTags \
--INPUT /dev/stdin \
--OUTPUT NA12878.recal.merged.new.t2t.sorted.sorted.bam \
--CREATE_INDEX true \
--CREATE_MD5_FILE true \
--REFERENCE_SEQUENCE ${ref_fasta} &

${gatk_path} --java-options "-Dsamjdk.compression_level=5 -Xms50000m -Djava.io.tmpdir=/czlab/xuwen/tmp" \
    SortSam \
--INPUT ${input_mapped_merged_marked_bam} \
--OUTPUT /dev/stdout \
--SORT_ORDER "coordinate" \
--CREATE_INDEX false \
--CREATE_MD5_FILE false \
| \
${gatk_path} --java-options "-Dsamjdk.compression_level=5 -Xms50000m -Djava.io.tmpdir=/czlab/xuwen/tmp" \
SetNmMdAndUqTags \
--INPUT /dev/stdin \
--OUTPUT NA12878.recal.merged.new.t2t.sorted.sorted_local.bam \
--CREATE_INDEX true \
--CREATE_MD5_FILE true \
--REFERENCE_SEQUENCE ${ref_fasta} &
