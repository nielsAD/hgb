#!/bin/bash

# Author:  Niels A.D.
# Project: HGB (https://github.com/nielsAD/hgb)
# License: Mozilla Public License, v2.0

set -e

export LOG=${LOG:-'generate.log'} # Output file with exit code of dataset processing
export GEN=${GEN:-'erdos_renyi regular powerlaw kronecker'}
export VRT=${VRT:-'1M'}
export EDG=${EDG:-'1 2 4 8 16 32 64 128'}
export RND=${RND:-'12345'}

export PP_ARG=${PP_ARG:-'-lqc'}  # graphsan options

export METIS_PARTS=${METIS_PARTS:-''}
export METIS_SEED=${METIS_SEED:-"$RND"}

generate_graph() {
	set -e

	local ALG="$1"
	local V="$2"
	local E="$3"
	local file="${ALG}/${ALG}_${V}_${E}"

	if [[ ! -s "${file}.gen.el_gz" ]] || [[ -n $FORCE ]]; then
		mkdir -p "$ALG"
		graphgen -a "$ALG" -r "$RND" -Oel_gz -m "$E" "$V" | graphsan $PP_ARG -Iel_gz - "${file}.gen.el_gz" > "${file}.log" 2>&1
	fi

	if [[ ! -s "${file}.gen.el_gz" ]] || [[ -n $FORCE ]]; then
		echo "Generation resulted in empty file; skipping." >> "${file}.log"
		[[ -z $CLEANUP ]] || rm -f "${file}.gen.el_gz"
		return
	fi

	if [[ ! -s "${file}.csr" ]] || [[ -n $FORCE ]]; then
		graphsan -dq "${file}.gen.el_gz" "${file}.csr" >> "${file}.log" 2>&1
		# graphchk "${file}.csr" >> "${file}.log" 2>&1
	fi

	echo "Partitioning for $METIS_PARTS" >> "${file}.log"
	for parts in $METIS_PARTS; do
		if [[ ! -s "${file}.gen.p${parts}.index" ]] || [[ -n $FORCE ]]; then
			gpmetis -seed="$METIS_SEED" "${file}.csr" "$parts" >> "${file}.log" 2>&1
			mv "${file}.csr.part.${parts}" "${file}.gen.p${parts}.index"
		fi
		if [[ ! -s "${file}.gen.p${parts}.cross" ]] || [[ -n $FORCE ]]; then
			graphpart "$parts" -mfile "${file}.gen.el_gz" >> "${file}.log" 2>&1
		fi
	done

	echo "Generated ${file} at $(date)" >> "${file}.log"
	[[ -z $CLEANUP ]] || rm "${file}.csr"
}

export -f generate_graph

export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1

parallel --lb --progress --eta -P4 --resume --joblog "$LOG" generate_graph ::: $GEN ::: $VRT ::: $EDG
