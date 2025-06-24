![Model](https://github.com/sefaakkoc/molytica/blob/main/img/Molytica2.png)
# Molytica (Suzuki-Miyaura Reaction Analyzer)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Mobile](https://img.shields.io/badge/Mobile-Responsive-purple.svg)]()

A comprehensive web-based analysis platform for analyzing, optimizing, and simulating Suzuki-Miyaura cross-coupling reactions.

## Features

### Web Interface (Flask)
- **Reaction Analysis**: Mechanistic analysis of Suzuki-Miyaura reactions
- **Catalyst Optimization**: Selection and optimization of palladium catalysts
- **Yield Calculation**: Theoretical vs experimental yield comparison
- **Side Product Analysis**: Identification and minimization of potential side products
- **Reaction Conditions**: Temperature, solvent, and base optimization
- **Molecular Visualization**: 2D/3D molecular structure rendering

### Mobile Application
- **Responsive Design**: Optimized experience across all devices
- **Offline Module**: Basic calculations without internet connection
- **QR Code Support**: Quick sharing of reaction data
- **Push Notifications**: Instant alerts for analysis results

## Requirements

### Backend (Flask)
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

### Frontend
```
HTML5/CSS3
JavaScript (ES6+)
Bootstrap 5
Chart.js
Three.js (molecular visualization)
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/username/suzuki-miyaura-analyzer.git
cd suzuki-miyaura-analyzer
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize Database
```bash
python init_db.py
```

### 5. Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`.

## Usage

### Web Interface

1. **Home Page**: Select reaction type (Aryl-Aryl, Aryl-Alkenyl, etc.)
2. **Reactant Input**: Enter reactants in SMILES format
3. **Catalyst Selection**: Choose appropriate Pd catalyst
4. **Condition Optimization**: Adjust temperature, solvent, base parameters
5. **Analysis**: View reaction mechanism and expected products

### API Usage

```python
import requests

# Reaction analysis
response = requests.post('http://localhost:5000/api/analyze', json={
    'aryl_halide': 'Cc1ccc(Br)cc1',
    'boronic_acid': 'c1ccc(B(O)O)cc1',
    'catalyst': 'Pd(PPh3)4',
    'base': 'K2CO3',
    'solvent': 'DMF',
    'temperature': 80
})

result = response.json()
print(f"Expected Yield: {result['yield']}%")
```

### Mobile Application

- Select "Quick Analysis" from main menu
- Scan reactants via QR code or enter manually
- View automatic condition suggestions
- Save results offline

## Project Structure

```
suzuki-miyaura-analyzer/
├── app.py                  # Main Flask application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── init_db.py            # Database initialization
├── models/
│   ├── __init__.py
│   ├── reaction.py       # Reaction models
│   └── catalyst.py       # Catalyst database
├── routes/
│   ├── __init__.py
│   ├── main.py          # Main routes
│   ├── api.py           # API endpoints
│   └── mobile.py        # Mobile APIs
├── static/
│   ├── css/
│   ├── js/
│   └── img/
├── templates/
│   ├── base.html
│   ├── index.html
│   └── mobile/
├── utils/
│   ├── chemistry.py     # Chemistry calculations
│   ├── visualization.py # Molecular visualization
│   └── optimizer.py     # Reaction optimization
├── mobile_app/          # Mobile app files
│   ├── manifest.json
│   ├── service-worker.js
│   └── pwa/
└── tests/
    ├── test_reactions.py
    └── test_api.py
```

## Scientific Foundation

This application is based on the following scientific principles:

- **Suzuki-Miyaura Mechanism**: Pd(0)/Pd(II) catalytic cycle
- **DFT Calculations**: Transition state energies
- **QSAR Models**: Reactivity predictions
- **Kinetic Analysis**: Rate constant calculations

## Supported Reaction Types

- Aryl-Aryl coupling
- Aryl-Alkenyl coupling  
- Heteroaryl coupling
- Alkenyl-Alkenyl coupling
- Alkyl-Aryl coupling (limited)

## API Endpoints

### Core Analysis
- `POST /api/analyze` - Complete reaction analysis
- `GET /api/catalysts` - Available catalyst list
- `POST /api/optimize` - Condition optimization
- `GET /api/predict/{smiles}` - Product prediction

### Mobile Specific
- `POST /api/mobile/quick-scan` - QR code analysis
- `GET /api/mobile/offline-data` - Offline calculation data
- `POST /api/mobile/save-result` - Save analysis results

## Mobile Features

### Progressive Web App (PWA)
- **Installable**: Add to home screen functionality
- **Offline Capable**: Core features work without internet
- **Fast Loading**: Optimized for mobile networks
- **Native Feel**: App-like experience on mobile devices

### Mobile-Specific Tools
- **Camera Integration**: Scan molecular structures
- **Touch Gestures**: Intuitive molecular manipulation
- **Responsive Charts**: Touch-friendly data visualization
- **Quick Actions**: Swipe gestures for common tasks

## Example Workflows

### Standard Analysis
1. Input aryl halide and boronic acid SMILES
2. Select catalyst from database
3. Choose reaction conditions
4. Run mechanistic analysis
5. Review yield predictions and side products

### Mobile Quick Analysis
1. Open mobile app
2. Scan QR code of substrate
3. Get instant condition recommendations
4. Save to favorites for later use

### Batch Processing
1. Upload CSV file with multiple reactions
2. Apply optimization algorithms
3. Export results as Excel/PDF report

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Create a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation for API changes
- Test mobile responsiveness

## Testing

### Run Tests
```bash
# Unit tests
python -m pytest tests/

# Chemistry validation
python tests/test_reactions.py

# API testing
python tests/test_api.py

# Mobile testing
npm test mobile_app/
```

### Coverage
```bash
pytest --cov=./ --cov-report=html
```

## Performance Metrics

- **Analysis Speed**: <2 seconds for standard reactions
- **Mobile Load Time**: <3 seconds on 3G networks
- **Database Queries**: Optimized for <100ms response
- **Concurrent Users**: Supports 100+ simultaneous analyses

## Security

- API rate limiting
- Input sanitization for SMILES strings
- Secure file upload handling
- HTTPS enforcement in production

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Lead Developer** - [GitHub](https://github.com/username)
- **Chemistry Advisor** - Assoc. Prof. dr Mitat Akkoc
- **Software Developer** - Sefa Akkoc

## Acknowledgments

- RDKit community for cheminformatics tools
- Flask development team
- Contributors to open-source chemistry software

## References

1. Miyaura, N.; Suzuki, A. *Chem. Rev.* **1995**, *95*, 2457-2483.
2. Martin, R.; Buchwald, S. L. *Acc. Chem. Res.* **2008**, *41*, 1461-1473.
3. Fortman, G. C.; Nolan, S. P. *Chem. Soc. Rev.* **2011**, *40*, 5151-5169.
4. Lennox, A. J. J.; Lloyd-Jones, G. C. *Chem. Soc. Rev.* **2014**, *43*, 412-443.

## Support

For issues and questions:
- Open an [Issue](https://github.com/username/suzuki-miyaura-analyzer/issues)
- Check the [Wiki](https://github.com/username/suzuki-miyaura-analyzer/wiki)
- Contact: support@suzuki-analyzer.com

## Roadmap

### Version 2.0 (Coming Soon)
- Machine learning yield prediction
- Advanced 3D visualization
- Multi-language support
- Cloud deployment options

### Future Features
- Integration with laboratory equipment
- Real-time collaboration tools
- Advanced statistical analysis
- Custom catalyst design tools

---

⚗**Note**: This software is intended for academic and research purposes. Please contact the developers before commercial use.

**Citation**: If you use this software in your research, please cite our paper: [DOI:10.xxxx/xxxxx]
