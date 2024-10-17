gatk_path=/homes8/xuwen/software/gatk/gatk
input_mapped_merged_marked_bam=/czlab/xuwen/results/NA12878/300X/outDir/NA12878/bams/NA12878.recal.merged.new.bam
ref_fasta=/czlab/xuwen/references/hg38/v0/Homo_sapiens_assembly38.fasta

srun --cpus-per-task=10 --mem=64G \
${gatk_path} --java-options "-Dsamjdk.compression_level=5 -Xms60000m -Djava.io.tmpdir=/czlab/xuwen/tmp" \
    SortSam \
--INPUT ${input_mapped_merged_marked_bam} \
--OUTPUT /dev/stdout \
--SORT_ORDER "coordinate" \
--CREATE_INDEX false \
--CREATE_MD5_FILE false \
| \
${gatk_path} --java-options "-Dsamjdk.compression_level=5 -Xms60000m -Djava.io.tmpdir=/czlab/xuwen/tmp" \
SetNmMdAndUqTags \
--INPUT /dev/stdin \
--OUTPUT NA12878.recal.merged.new.sorted.bam \
--CREATE_INDEX true \
--CREATE_MD5_FILE true \
--REFERENCE_SEQUENCE ${ref_fasta}
