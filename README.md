![Model](https://github.com/sefaakkoc/molytica/blob/main/img/Molytica2.png)
# Molytica (Suzuki-Miyaura Reaction Analyzer)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive web-based analysis platform for analyzing, optimizing, and simulating Suzuki-Miyaura cross-coupling reactions.

## Features

- **Reaction Analysis**: Mechanistic analysis of Suzuki-Miyaura reactions
- **Catalyst Optimization**: Selection and optimization of palladium catalysts
- **Yield Calculation**: Theoretical vs experimental yield comparison
- **Side Product Analysis**: Identification and minimization of potential side products
- **Reaction Conditions**: Temperature, solvent, and base optimization
- **Molecular Visualization**: 2D/3D molecular structure rendering
- **Responsive Design**: Optimized experience across all devices

## Requirements

```
Python >= 3.8
Flask >= 2.0
RDKit >= 2022.03
NumPy >= 1.21
Pandas >= 1.3
Matplotlib >= 3.5
Plotly >= 5.0
SQLAlchemy >= 1.4
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sefaakkoc/molytica.git
cd molytica
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python run.py
```

The application will be available at `http://0.0.0.0:5000`.

## Usage

### Web Interface

1. **Home Page**: Select reaction type (Aryl-Aryl, Aryl-Alkenyl, etc.)
2. **Reactant Input**: Enter reactants in SMILES format
3. **Catalyst Selection**: Choose appropriate Pd catalyst
4. **Condition Optimization**: Adjust temperature, solvent, base parameters
5. **Analysis**: View reaction mechanism and expected products

## Project Structure

```
molytica/
├───code
├───outputs
├───static
│   ├───assets
│   ├───datasets
│   ├───images
│   ├───model
│   ├───outputs
│   └───xml_data
└───templates
```

## Scientific Foundation

This application is based on the following scientific principles:

- **Suzuki-Miyaura Mechanism**: Pd(0)/Pd(II) catalytic cycle
- **DFT Calculations**: Transition state energies
- **QSAR Models**: Reactivity predictions
- **Kinetic Analysis**: Rate constant calculations

## Supported Reaction Types

- Aryl-Aryl coupling
- Alkyl-Aryl coupling (limited)

## Example Workflows

### Standard Analysis
1. Input aryl halide and boronic acid SMILES
2. Select catalyst from database
3. Choose reaction conditions
4. Run mechanistic analysis
5. Review yield predictions and side products

### Batch Processing
1. Upload CSV file with multiple reactions
2. Apply optimization algorithms
3. Export results as Excel/PDF report

## Dataset Requirements

### CSV File Format
For batch analysis, your CSV file should contain the following columns:

| Column Name | Description | Example |
|------------|-------------|---------|
| `Ar-B(OH)2` | Boronic acid compound name | `phenylboronic acid` |
| `Ar-X` | Aryl halide compound name | `bromobenzene` |
| `product` | Expected product name | `1,1'-biphenyl` |
| `catalizor` | Catalyst SMILES notation | `I[Pd](I)([N]1=CC=CC=C1)C(N2C)N(C)C3=C2N(C)C(N(C)C3=O)=O` |
| `base` | Base SMILES notation | `[K].[K]OO[C]=O` |
| `solv1` | Primary solvent | `water` |
| `solv2` | Secondary solvent | `propan-2-ol` |
| `amount` | Catalyst amount (mol) | `0.0025` |
| `centigrades` | Temperature in °C | `40` |
| `minute` | Reaction time in minutes | `120` |
| `cycle` | Reaction cycle number | `88` |
| `yield` | Experimental yield (%) | `81` |

### Sample CSV Structure
```csv
Ar-B(OH)2,Ar-X,product,catalizor,base,solv1,solv2,amount,centigrades,minute,cycle,yield
phenylboronic acid,bromobenzene,1,1'-biphenyl,I[Pd](I)([N]1=CC=CC=C1)C(N2C)N(C)C3=C2N(C)C(N(C)C3=O)=O,[K].[K]OO[C]=O,water,propan-2-ol,0.0025,40,120,88,81
4-methylphenylboronic acid,4-bromobenzaldehyde,4'-methyl-[1,1'-biphenyl]-4-carbaldehyde,I[Pd](I)([N]1=CC=CC=C1)C(N2C)N(C)C3=C2N(C)C(N(C)C3=O)=O,[K].[K]OO[C]=O,water,ethanol,0.003,60,180,92,78
```

### Analysis Process

- Analysis may take **2-5 minutes** per reaction depending on complexity
- Large datasets (>50 reactions) may require **10-30 minutes** of processing time
- Do not close the browser window during analysis
- Results will be displayed progressively as each reaction completes
- Failed analyses will be marked with error messages for troubleshooting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Lead Developer** - [Sefa Akkoc](https://github.com/sefaakkoc)
- **Chemistry Advisor** - Assoc. Prof. Dr. Mitat Akkoc

## References

1. Miyaura, N.; Suzuki, A. *Chem. Rev.* **1995**, *95*, 2457-2483.
2. Martin, R.; Buchwald, S. L. *Acc. Chem. Res.* **2008**, *41*, 1461-1473.
3. Fortman, G. C.; Nolan, S. P. *Chem. Soc. Rev.* **2011**, *40*, 5151-5169.
4. Lennox, A. J. J.; Lloyd-Jones, G. C. *Chem. Soc. Rev.* **2014**, *43*, 412-433.

## Support

For issues and questions:
- Open an [Issue](https://github.com/sefaakkoc/molytica/issues)
- Check the [Wiki](https://github.com/sefaakkoc/molytica/wiki)

**Note**: This software is intended for academic and research purposes. Please contact the developers before commercial use.
