from flask import Flask, render_template, request, jsonify, Response, send_file
import os
import logging
import subprocess
import threading
import glob
from datetime import datetime
import pandas as pd
import re
import requests
from urllib.parse import urlparse

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
"""sa"""
rowcount = 0
DATASET_LOG_DIR = 'datasets_log'
DATASET_LOG_DIR2 = 'static/datasets'
CODE_LOG_DIR = 'code'
IMAGES_LOG_DIR = 'static/images'
RESULTS_DIR = 'static/outputs' 
TRAINED_MODELS_DIR = 'trained' 
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'static/datasets')

def merge_csv_to_output(input_files, output_file='out.csv'):
    global rowcount
    try:
        a=[]
        for x in input_files:
            a.append("static/datasets/"+x)
            
        dfs = [pd.read_csv(f) for f in a]
        merged_df = pd.concat(dfs, ignore_index=True)
        rowcount = len(merged_df)
        merged_df.to_csv(output_file, index=False)
        asdf=os.path.join('static/model', 'miaw.csv')
        merged_df.to_csv(asdf, index=False)

        return rowcount

    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return 0

def get_rowcount_for_datasets(dataset_files):
    """Seçilen dataset dosyalarının toplam satır sayısını hesapla"""
    try:
        total_rows = 0
        for dataset_file in dataset_files:
            file_path = os.path.join("static/datasets", dataset_file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                total_rows += len(df)
        return total_rows
    except Exception as e:
        print(f"Satır sayısı hesaplanırken hata: {str(e)}")
        return 0

@app.route('/predict')
def predict():
    global rowcount
    datasets = get_datasets_from_folder(DATASET_LOG_DIR2)
    codess = get_codes_from_folder(CODE_LOG_DIR)
    app.logger.info("Accessed the main index page.")
    return render_template('predict.html', datasets=datasets, codess=codess, rowcount=rowcount)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/get_dataset_rowcount', methods=['POST'])
def get_dataset_rowcount():
    """Seçilen dataset'lerin toplam satır sayısını döndür"""
    data = request.get_json()
    selected_datasets = data.get('selectedDatasets', [])
    
    if not selected_datasets:
        return jsonify({"rowcount": 0})
    
    total_rows = get_rowcount_for_datasets(selected_datasets)
    return jsonify({"rowcount": total_rows})

@app.route('/visual')
def visual():
    folders = [f.name for f in os.scandir(IMAGES_LOG_DIR) if f.is_dir()]
    return render_template('visual.html', images=folders)

@app.route('/run_model', methods=['POST'])
def run_model():
    data = request.get_json()
    selected_datasets = data.get('selectedDatasets', [])
    selected_code = data.get('selectedCode', '')
    
   
    global rowcount
    rowcount = merge_csv_to_output(selected_datasets,"out.csv")

    if not selected_code:
        return jsonify({"status": "error", "message": "No model script selected."}), 400

    app.logger.info(f"Starting model: {selected_code} with datasets: {selected_datasets}")

   
    thread = threading.Thread(target=run_python_script, args=(selected_code, selected_datasets))
    thread.start()

    return jsonify({"status": "success", "message": "Model is running!"})

@app.route('/stream_output')
def stream_output():
    selected_code = request.args.get('code')
    
    def generate():
        if not selected_code:
            yield "data: ERROR: No script selected\n\n"
            return

        command = ['python', f'code/{selected_code}']
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

       
        while True:
            output = process.stdout.readline()
            if output:
                yield f"data: {output.strip()}\n\n"
            
            error = process.stderr.readline()
            if error:
                yield f"data: ERROR: {error.strip()}\n\n"
            
            if process.poll() is not None: 
               
                for output in process.stdout:
                    if output: yield f"data: {output.strip()}\n\n"
                for error in process.stderr:
                    if error: yield f"data: ERROR: {error.strip()}\n\n"
                yield "data: COMPLETE: Model training finished\n\n"
                break

    return Response(generate(), content_type='text/event-stream')

@app.route('/get_latest_result')
def get_latest_result():
    """Get the latest result file content"""
    try:
       
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
       
        result_files = glob.glob(os.path.join(RESULTS_DIR, '*.txt'))
        if not result_files:
            return jsonify({"status": "error", "message": "No results found"}), 404
        
        latest_file = max(result_files, key=os.path.getctime)
        
        with open(latest_file, 'r') as f:
            content = f.read()
            
        return jsonify({
            "status": "success",
            "content": content,
            "filename": os.path.basename(latest_file)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/download_model')
def download_model():
    """Download the trained model file"""
    model_path = os.path.join(TRAINED_MODELS_DIR, 'models.pkl')
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    return jsonify({"status": "error", "message": "Model file not found"}), 404


@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/edit_dataset')
def edit_dataset():
    """Render the edit dataset page"""
    return render_template('edit.html')


@app.route('/preview_dataset')
def preview_dataset():
    """Render the preview dataset page"""
    return render_template('preview.html')

@app.route('/excel_to_csv')
def excel_to_csv():
    """Render the excel to csv conversion page"""
    return render_template('xlsx.html')



@app.route('/manual')
def manuel():
    return render_template('manual.html')

@app.route('/upload-csv')
def upload_csv():
    return render_template('csv.html')

@app.route('/save_data', methods=['POST'])
def save_data():
    try:
        data = request.get_json()
        filename = data.get('filename', 'data')
        content = data.get('data', '')
        
       
        os.makedirs('static/datasets', exist_ok=True)
        
       
        filepath = os.path.join('static/datasets', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return jsonify({'success': True, 'message': 'File saved successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/compare-models')
def compare_models():
    return render_template('messi.html')



@app.route('/get_xml_files')
def get_xml_files():
    """Klasördeki XML dosyalarını listeler"""
    xml_files = []
    try:
       
        xml_dir = os.path.join(app.root_path, 'xml_data')
        os.makedirs(xml_dir, exist_ok=True)
        
        for file in glob.glob(os.path.join(xml_dir, '*.xml')):
            xml_files.append(os.path.basename(file))
    except Exception as e:
        app.logger.error(f"XML dosyaları listelenirken hata: {str(e)}")
    
    return jsonify(xml_files)

@app.route('/get_xml_data')
def get_xml_data():
    """Seçilen XML dosyasının içeriğini döndürür"""
    filename = request.args.get('file', '')
    if not filename:
        return jsonify({"error": "Dosya adı belirtilmedi"}), 400
    
    try:
       
        if not filename.lower().endswith('.xml'):
            return jsonify({"error": "Geçersiz dosya türü"}), 400
            
       
        filepath = os.path.join(app.root_path, 'xml_data', filename)
        
       
        if not os.path.exists(filepath):
            return jsonify({"error": "Dosya bulunamadı"}), 404
            
       
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return content, 200, {'Content-Type': 'text/xml'}
    except Exception as e:
        app.logger.error(f"XML dosyası okunurken hata: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/save-csv', methods=['POST'])
def save_csv():
    try:
        data = request.get_json()
        csv_content = data.get('csv', '')
        filenamea = data.get('filename', '')
        
        if not csv_content:
            return jsonify({'success': False, 'message': 'Boş CSV içeriği'})
        
        save_path = os.path.join(DATASET_LOG_DIR2, filenamea)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)
            
        return jsonify({
            'success': True,
            'message': f'CSV başarıyla kaydedildi: {filenamea}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Hata oluştu: {str(e)}'
        })

@app.route('/api/load-csv', methods=['POST'])
def api_load_csv():
    """API endpoint to load CSV from URL"""
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({'success': False, 'message': 'URL is required'}), 400
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return jsonify({
            'success': True,
            'content': response.text,
            'filename': os.path.basename(urlparse(url).path)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/save-csv', methods=['POST'])
def api_save_csv():
    """API endpoint to save CSV data"""
    try:
        data = request.get_json()
        content = data.get('content', '')
        filename = data.get('filename', '')
        
        if not content:
            return jsonify({'success': False, 'message': 'No content provided'}), 400
        
       
        filename = sanitize_csv_filename(filename)
        print(filename)
        
       
        os.makedirs(DATASET_LOG_DIR2, exist_ok=True)
        
       
        filepath = os.path.join(DATASET_LOG_DIR2, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return jsonify({
            'success': True,
            'message': 'CSV saved successfully',
            'filename': filename,
            'path': f'static/datasets/{filename}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def sanitize_csv_filename(filename):
    """Sanitize the filename for CSV files"""
   
    filename = os.path.basename(filename)
   
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
   
    filename = re.sub(r'[^\w\-_.]', '_', filename)
    return filename

def run_python_script(script_name, datasets):
    command = ['python', f'code/{script_name}'] + datasets
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

   
    for line in process.stdout:
        app.logger.info(line.strip())

    process.wait()

def get_datasets_from_folder(folder_path):
    datasets = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    datasets = [f for f in datasets if f.endswith('.csv')]
    return datasets

def get_codes_from_folder(folder_path):
    codes = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    codes = [f for f in codes if f.endswith('.py')]
    return codes

if __name__ == '__main__':
   
    os.makedirs(DATASET_LOG_DIR, exist_ok=True)
    os.makedirs(CODE_LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_LOG_DIR, exist_ok=True)
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000,debug=True)
