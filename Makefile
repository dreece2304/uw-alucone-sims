.PHONY: env update activate demo pnnl-cli adapters-cli test run-pca phase02 phase03 phase04 status validate-registry

env:
	micromamba env create -n sims-pca -f env/environment.yml

update:
	micromamba env update -n sims-pca -f env/environment.yml

activate:
	@echo "micromamba activate sims-pca"

demo:
	micromamba run -n sims-pca python -V
	micromamba run -n sims-pca pip list | grep -E 'scikit-learn|python-docx'

pnnl-cli:
	micromamba run -n sims-pca python ATOFSIMSCLASS/SIMS_PCA/SIMS_PCA/src/main.py --help || true

adapters-cli:
	micromamba run -n sims-pca python -m adapters.cli --help

test:
	micromamba run -n sims-pca python -c "import adapters; print('✓ Adapters package works')"
	micromamba run -n sims-pca python -m adapters.cli --help

# Run PNNL PCA analysis headless
# Usage: make run-pca RAW=path/to/raw.tsv CATALOG=path/to/catalog.csv DOC=path/to/doc_mass.csv ION=P OUT=results_run1
run-pca:
	$(if $(RAW),,$(error RAW is required. Usage: make run-pca RAW=... CATALOG=... DOC=... ION=P OUT=...))
	$(if $(CATALOG),,$(error CATALOG is required. Usage: make run-pca RAW=... CATALOG=... DOC=... ION=P OUT=...))
	$(if $(DOC),,$(error DOC is required. Usage: make run-pca RAW=... CATALOG=... DOC=... ION=P OUT=...))
	$(if $(ION),,$(error ION is required (P or N). Usage: make run-pca RAW=... CATALOG=... DOC=... ION=P OUT=...))
	$(if $(OUT),,$(error OUT is required. Usage: make run-pca RAW=... CATALOG=... DOC=... ION=P OUT=...))
	micromamba run -n sims-pca python runner/run_pnnl_pca.py \
		--pca-dir ATOFSIMSCLASS/SIMS_PCA/SIMS_PCA \
		--out-dir $(OUT) \
		--raw $(RAW) \
		--catalog $(CATALOG) \
		--doc-mass $(DOC) \
		--ion $(ION) \
		--top-n 5

phase02:
	@echo "Run Phase 02 only after contracts are satisfied"
phase03:
	@echo "Run Phase 03 only after contracts are satisfied"
phase04:
	@echo "Run Phase 04 only after contracts are satisfied"
status:
	@cat _shared/state/STATUS.json || true

validate-registry:
	./scripts/validate_registry.py || true