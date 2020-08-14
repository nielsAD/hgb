#!/bin/bash

# Author:  Niels A.D.
# Project: HGB (https://github.com/nielsAD/hgb)
# License: Mozilla Public License, v2.0

set -e

export LOG=${LOG:-'download.log'} # Output file with exit code of dataset processing

export PP_ARG=${PP_ARG:-'-lqc'}  # graphsan options

export METIS_PARTS=${METIS_PARTS:-'2 4 8 16'}
export METIS_SEED=${METIS_SEED:-'12345'}

export UFL_PRE=${UFL_PRE:-'.'}                                                                    # Prefix for output files
export UFL_URL=${UFL_URL:-'https://www.cise.ufl.edu/research/sparse/matrices/list_by_id.html'}
export UFL_SUB=${UFL_SUB:-'DIMACS10|SNAP|Gleich|Law'}                                             # Extended grep patterns to incl (e.g. 'SNAP|Newman|Grund|GSet|Nasa|JGD_Forest|JGD_Trefethen|JGD_Homology|JGD_Relat' or '*')
export UFL_BAN=${UFL_BAN:-'as-735|as-caida|2010|vsp_|t2d_|uk-2005|sk-2005|it-2004|indochina-2004|webbase-2001'}  # Extended grep patterns to excl (e.g. 'as-735|as-caida' or '^$')
export UFL_SRC=${UFL_SRC:-'ufl.txt'}                                                              # Output file with list of downloaded datasets

url_to_filename() {
	local file=$(echo "$1" | grep -o '/MM/.*' | cut -f3- -d'/')
	echo "${UFL_PRE}/${file}"
}

is_sparse_matrix() {
	head -n 1 "$1" | grep -iw 'coordinate'
}

is_symmetric() {
	head -n 1 "$1" | grep -Eiw 'symmetric|skew-symmetric|hermitian'
}

download_matrix() {
	set -e
	local name=$(url_to_filename "$1")
	local dir=$(dirname "$name")
	mkdir -p "$dir"
	wget -q -N -c -O "$name" "$1"
}

unpack_matrix() {
	set -e
	local name=$(url_to_filename "$1")
	local dir=$(dirname "$name")
	local ts=$(date)
	tar --keep-newer-files --warning=none -xf "$name" -C "$dir"
	tar -tf "$name" | while read line; do
		local file="${dir}/${line}"
		if [[ -d "$file" ]]; then
			# Skip directories
			continue
		fi

		if [[ -z $(is_sparse_matrix "$file") ]] 2>/dev/null; then
			# Not a valid matrix
			[[ -z $CLEANUP ]] || rm "$file"
			continue
		fi

		echo "Downloaded and extracted from $1 at $ts" >> "${file}.log"

		if sha512sum -c --quiet "${file}.sha512"  2>/dev/null; then
			echo "Nothing has changed since last run; skipping." >> "${file}.log"
			[[ -z $CLEANUP ]] || rm "$file"
			continue
		fi

		local fmt; if [[ $(is_symmetric "$file") ]]; then fmt="uel"; else fmt="del"; fi
		(grep -v '^%' "$file" | tail -n +2 | graphsan $PP_ARG -I"$fmt" - "${file}.el_gz") >> "${file}.log" 2>&1

		if ! [[ -s "${file}.el_gz" ]]; then
			echo "Cleaning resulted in empty file; skipping." >> "${file}.log"
			[[ -z $CLEANUP ]] || rm -f "${file}" "${file}.el_gz"
			continue
		fi

		sha512sum "${file}.el_gz" > "${file}.sha512"

		graphsan -dq "${file}.el_gz" "${file}.csr" >> "${file}.log" 2>&1
		#graphchk "${file}.csr" >> "${file}.log" 2>&1

		echo "Partitioning for $METIS_PARTS" >> "${file}.log"
		for parts in $METIS_PARTS; do
			gpmetis -seed="$METIS_SEED" "${file}.csr" "$parts" >> "${file}.log" 2>&1
			mv "${file}.csr.part.${parts}" "${file}.p${parts}.index"
			sha512sum "${file}.p${parts}.index" >> "${file}.sha512"
			#graphpart "$parts" -mfile "${file}.el_gz" >> "${file}.log" 2>&1
		done

		echo "Processed $file at $(date)" >> "${file}.log"
		[[ -z $CLEANUP ]] || rm "$file" "${file}.csr"
	done
	[[ -z $CLEANUP ]] || rm "$name"
}

export -f url_to_filename
export -f is_symmetric
export -f is_sparse_matrix

export -f download_matrix
export -f unpack_matrix

export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1

lynx -listonly -dump "$UFL_URL" | grep -Eo 'https?:.*' | grep "/MM/" | grep -Ei "$UFL_SUB" | grep -vEi "$UFL_BAN" > "$UFL_SRC"
parallel --lb --progress --eta -P32                download_matrix :::: "$UFL_SRC"
parallel --lb --progress --eta -P4 --joblog "$LOG" unpack_matrix   :::: "$UFL_SRC"
