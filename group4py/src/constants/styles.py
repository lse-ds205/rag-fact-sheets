
EMAIL_CSS_STYLE = """
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f8f9fa;
                padding: 20px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.2em;
                font-weight: 700;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .header .subtitle {
                font-size: 1.1em;
                opacity: 0.9;
                font-weight: 300;
            }
            
            .metadata {
                background-color: #f1f3f4;
                border-left: 5px solid #667eea;
                padding: 25px;
                margin: 0;
            }
            
            .metadata-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
            }
            
            .metadata-item {
                background-color: white;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #e1e5e9;
            }
            
            .metadata-label {
                font-weight: 600;
                color: #495057;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 5px;
            }
            
            .metadata-value {
                font-size: 1.1em;
                color: #212529;
                font-weight: 500;
            }
            
            .target-years {
                display: inline-flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            
            .target-year {
                background-color: #667eea;
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: 500;
            }
            
            .content {
                padding: 30px;
            }
            
            .section-title {
                font-size: 1.8em;
                color: #495057;
                margin-bottom: 25px;
                padding-bottom: 10px;
                border-bottom: 3px solid #667eea;
                font-weight: 600;
            }
            
            .qa-container {
                margin-bottom: 30px;
            }
            
            .question {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-left: 4px solid #28a745;
                padding: 18px;
                margin-bottom: 15px;
                border-radius: 0 6px 6px 0;
                font-weight: 600;
                color: #495057;
                font-size: 1.05em;
            }
            
            .answer {
                background-color: #fff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 20px;
                margin-bottom: 25px;
                line-height: 1.7;
                color: #212529;
            }
            
            .answer p {
                margin-bottom: 10px;
            }
            
            .no-data {
                color: #6c757d;
                font-style: italic;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 4px;
                text-align: center;
            }
            
            .footer {
                background-color: #495057;
                color: white;
                text-align: center;
                padding: 20px;
                font-size: 0.9em;
            }
            
            @media print {
                body {
                    background-color: white;
                    padding: 0;
                }
                
                .container {
                    box-shadow: none;
                    border-radius: 0;
                }
                
                .header {
                    background: #667eea !important;
                    -webkit-print-color-adjust: exact;
                }
                
                .target-year {
                    background-color: #667eea !important;
                    -webkit-print-color-adjust: exact;
                }
            }
        </style>
        """