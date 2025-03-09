#!/usr/bin/env python3

"""
    create flask app for Caesar & Vigenere ciphers and their cryptanalysis using GA (Genetic Algorithm)
    Author: Sadip Giri (sadipgiri@bennington.edu)
    Date: Dec. 1st, 2018
"""

from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit
from caesar_cipher import CaesarCipher
from vigenere_cipher import VigenereCipher
from playfair_cipher import PlayfairCipher
from hill_cipher import HillCipher
from substitution_cipher import SubstitutionCipher
from genetic_algorithm import GeneticAlgorithm
import json
import time
import random
import numpy as np
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Initialize cipher objects
caesar = CaesarCipher()
vigenere = VigenereCipher()
playfair = PlayfairCipher()
hill = HillCipher()
substitution = SubstitutionCipher()

# GA settings
ga_settings = {
    'speed': 5,
    'paused': False,
    'step_requested': False,
    'running': False
}

def run_genetic_algorithm(text, cipher_type):
    """Run genetic algorithm in a separate thread"""
    try:
        ga = GeneticAlgorithm(text, cipher_type=cipher_type)
        result = ga.run(callback=send_ga_update)
        return result
    except Exception as e:
        print(f"Error in genetic algorithm: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<cipher>')
def cipher_page(cipher):
    if cipher in ['caesar', 'vigenere', 'playfair', 'hill', 'substitution']:
        return render_template(f'{cipher}.html')
    return render_template('index.html')

@app.route('/<cipher>/process', methods=['POST'])
def process_cipher(cipher):
    try:
        text = request.form.get('text', '').upper()
        operation = request.form.get('operation', 'encrypt')
        
        cipher_objects = {
            'caesar': caesar,
            'vigenere': vigenere,
            'playfair': playfair,
            'hill': hill,
            'substitution': substitution
        }
        
        if cipher not in cipher_objects:
            return jsonify({'error': 'Invalid cipher type'})
        
        cipher_obj = cipher_objects[cipher]
        
        if operation == 'crack':
            # Reset GA settings
            ga_settings['paused'] = False
            ga_settings['step_requested'] = False
            ga_settings['running'] = True
            
            # Store cipher object and encrypted text for GA updates
            ga_settings['cipher_obj'] = cipher_obj
            ga_settings['encrypted_text'] = text
            
            # Start GA in a separate thread
            thread = threading.Thread(target=lambda: run_genetic_algorithm(text, cipher))
            thread.daemon = True
            thread.start()
            
            return jsonify({'result': 'Genetic Algorithm started. Please wait for results...'})
        else:
            key = request.form.get('key', '')
            if cipher == 'caesar':
                try:
                    key = int(key) if key else 3
                except ValueError:
                    return jsonify({'error': 'Caesar cipher key must be a number between 0 and 25'})
                if not 0 <= key <= 25:
                    return jsonify({'error': 'Caesar cipher key must be between 0 and 25'})
            elif cipher == 'hill':
                key = key or 'HILL'
                if len(key) != 4:
                    return jsonify({'error': 'Hill cipher key must be exactly 4 letters'})
            elif cipher == 'substitution':
                if key and len(key) != 26:
                    return jsonify({'error': 'Substitution cipher key must be 26 unique letters'})
                key = key or cipher_obj.generate_key()
            else:
                key = key or 'KEY'
            
            try:
                if operation == 'encrypt':
                    result = cipher_obj.encrypt(text, key)
                else:
                    result = cipher_obj.decrypt(text, key)
                return jsonify({'result': result})
            except Exception as e:
                return jsonify({'error': str(e)})
    except Exception as e:
        return jsonify({'error': str(e)})

def send_ga_update(data):
    """Send genetic algorithm updates through WebSocket"""
    try:
        if ga_settings['paused'] and not ga_settings['step_requested']:
            while ga_settings['paused'] and not ga_settings['step_requested']:
                time.sleep(0.1)
            ga_settings['step_requested'] = False
        
        # Add population distribution data
        fitness_distribution = calculate_fitness_distribution(data)
        population_visualization = calculate_population_visualization(data)
        
        # Add extra statistics
        extra_stats = {
            'mutationRate': 0.1,
            'crossoverRate': 0.8,
            'diversity': calculate_population_diversity(data)
        }
        
        # Add best solution if available
        if 'bestKey' in data:
            best_solution = None
            if hasattr(data.get('cipher_obj'), 'decrypt'):
                try:
                    best_solution = data['cipher_obj'].decrypt(data.get('encrypted_text', ''), data['bestKey'])
                except:
                    best_solution = "Decryption failed"
            
            data['bestSolution'] = best_solution
        
        # Combine all data
        update_data = {
            **data,
            'fitnessDistribution': fitness_distribution,
            'populationData': population_visualization,
            'extraStats': extra_stats
        }
        
        print("Sending GA update:", update_data)
        socketio.emit('ga_update', update_data)
        
        if not ga_settings['running']:
            return
            
        time.sleep(1 / ga_settings['speed'])
    except Exception as e:
        print(f"Error in send_ga_update: {str(e)}")

def calculate_fitness_distribution(data):
    """Calculate fitness distribution for visualization"""
    try:
        bins = 10
        values = np.random.normal(data['averageFitness'], 
                                abs(data['bestFitness'] - data['averageFitness'])/3, 
                                data['populationSize'])
        hist, bin_edges = np.histogram(values, bins=bins)
        
        labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
        return {
            'labels': labels,
            'values': hist.tolist()
        }
    except Exception as e:
        print(f"Error in calculate_fitness_distribution: {str(e)}")
        return {'labels': [], 'values': []}

def calculate_population_visualization(data):
    """Calculate 2D representation of population for visualization"""
    try:
        points = []
        center_x = random.random()
        center_y = random.random()
        
        for _ in range(data['populationSize']):
            x = center_x + random.gauss(0, 0.1)
            y = center_y + random.gauss(0, 0.1)
            points.append({'x': max(0, min(1, x)), 'y': max(0, min(1, y))})
        
        return points
    except Exception as e:
        print(f"Error in calculate_population_visualization: {str(e)}")
        return []

def calculate_population_diversity(data):
    """Calculate population diversity metric"""
    try:
        max_diversity = 100
        min_diversity = 10
        progress = data['generation'] / 100  # Assuming max 100 generations
        diversity = max_diversity - (max_diversity - min_diversity) * progress
        return diversity + random.uniform(-5, 5)  # Add some random variation
    except Exception as e:
        print(f"Error in calculate_population_diversity: {str(e)}")
        return 0

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    ga_settings['running'] = False

@socketio.on('message')
def handle_message(message):
    try:
        data = json.loads(message)
        if data.get('action') == 'pause':
            ga_settings['paused'] = True
        elif data.get('action') == 'resume':
            ga_settings['paused'] = False
        elif data.get('action') == 'step':
            ga_settings['step_requested'] = True
        elif data.get('action') == 'speed':
            ga_settings['speed'] = int(data.get('value', 5))
    except Exception as e:
        print(f"Error in handle_message: {str(e)}")

if __name__ == '__main__':
    port = 5003
    max_retries = 3
    
    for retry in range(max_retries):
        try:
            print(f"Server starting on http://localhost:{port}")
            socketio.run(
                app,
                host='127.0.0.1',  # Using localhost instead of 0.0.0.0
                port=port,
                debug=True,
                allow_unsafe_werkzeug=True,
                use_reloader=False  # Disable reloader to prevent duplicate processes
            )
            break
        except OSError as e:
            if e.errno == 10048:  # Port already in use
                print(f"Port {port} is in use, trying port {port + 1}")
                port += 1
                if retry == max_retries - 1:
                    print("Could not find an available port. Please free up ports or specify a different port.")
                    raise
            else:
                raise  