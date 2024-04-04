awk '{if($3 == "gene") {print $9}}' /home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/MANE/v1.3/MANE.GRCh38.v1.3.refseq_genomic.gff | awk '{
    n = split($0, array, ";");  # Split the line into parts based on ";"
    for (i = 1; i <= n; i++) {
        if (array[i] ~ /^ID=/) {  # Look for the part that starts with "ID="
            split(array[i], idArray, "=");  # Split that part into key and value based on "="
            print idArray[2];  # Print the value part
            break;  # Exit the loop after finding and printing the ID
        }
    }
}'  | sort 