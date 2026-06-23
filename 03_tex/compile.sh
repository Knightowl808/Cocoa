#!/bin/bash
# =============================================================================
# LaTeX Thesis Compilation Script
# =============================================================================
#
# Usage:
#   ./compile.sh           # Full build (pdflatex + biber + pdflatex x2)
#   ./compile.sh quick     # Quick build (pdflatex only, no bibliography)
#   ./compile.sh clean     # Clean auxiliary files
#   ./compile.sh watch     # Watch mode (requires latexmk)
#
# =============================================================================

set -e  # Exit on error

# Configuration
MAIN_FILE="main"
OUTPUT_DIR="."
BUILD_DIR="build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if LaTeX is installed
check_latex() {
    if ! command -v pdflatex &> /dev/null; then
        print_error "pdflatex not found. Please install TeX Live first."
        echo ""
        echo "On Ubuntu/Debian, run:"
        echo "  sudo apt-get install texlive-latex-base texlive-latex-recommended \\"
        echo "       texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra biber latexmk"
        echo ""
        exit 1
    fi
}

# Full build with bibliography
full_build() {
    print_status "Starting full build..."

    print_status "Step 1/4: First pdflatex pass..."
    pdflatex -interaction=nonstopmode -halt-on-error ${MAIN_FILE}.tex

    print_status "Step 2/4: Running biber for bibliography..."
    if command -v biber &> /dev/null; then
        biber ${MAIN_FILE} || print_warning "Biber failed (bibliography may be incomplete)"
    else
        print_warning "biber not found, skipping bibliography"
    fi

    print_status "Step 3/4: Second pdflatex pass..."
    pdflatex -interaction=nonstopmode -halt-on-error ${MAIN_FILE}.tex

    print_status "Step 4/4: Third pdflatex pass (for cross-references)..."
    pdflatex -interaction=nonstopmode -halt-on-error ${MAIN_FILE}.tex

    print_status "Build complete! Output: ${MAIN_FILE}.pdf"
}

# Quick build (no bibliography)
quick_build() {
    print_status "Starting quick build (no bibliography)..."
    pdflatex -interaction=nonstopmode -halt-on-error ${MAIN_FILE}.tex
    print_status "Quick build complete! Output: ${MAIN_FILE}.pdf"
}

# Watch mode using latexmk
watch_build() {
    if ! command -v latexmk &> /dev/null; then
        print_error "latexmk not found. Install it with: sudo apt-get install latexmk"
        exit 1
    fi

    print_status "Starting watch mode (Ctrl+C to stop)..."
    latexmk -pdf -pvc -interaction=nonstopmode ${MAIN_FILE}.tex
}

# Clean auxiliary files
clean_files() {
    print_status "Cleaning auxiliary files..."

    # List of extensions to clean
    EXTENSIONS="aux bbl bcf blg fdb_latexmk fls log out run.xml toc lof lot synctex.gz"

    for ext in $EXTENSIONS; do
        if ls *.${ext} 1> /dev/null 2>&1; then
            rm -f *.${ext}
            print_status "Removed *.${ext}"
        fi
    done

    print_status "Clean complete!"
}

# Main script
check_latex

case "${1:-full}" in
    full)
        full_build
        ;;
    quick)
        quick_build
        ;;
    watch)
        watch_build
        ;;
    clean)
        clean_files
        ;;
    *)
        echo "Usage: $0 {full|quick|watch|clean}"
        echo ""
        echo "  full   - Full build with bibliography (default)"
        echo "  quick  - Quick build without bibliography"
        echo "  watch  - Watch mode with auto-recompile"
        echo "  clean  - Remove auxiliary files"
        exit 1
        ;;
esac
