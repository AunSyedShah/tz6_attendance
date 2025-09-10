# AI Attendance System - Django Version

A comprehensive student enrollment system converted from Streamlit to Django, featuring real-time face detection, image capture, and progress tracking.

## 🚀 Features

### Core Functionality
- **Student Management**: Create, view, and manage student profiles
- **Live Camera Enrollment**: Real-time face detection and image capture
- **Demo Mode**: Upload multiple images for testing without camera access
- **Progress Tracking**: Real-time enrollment progress with visual indicators
- **Quality Validation**: Automatic face detection with quality metrics
- **Interactive Prompts**: Guided image capture with 10 different poses/expressions

### Technical Features
- **Face Detection**: OpenCV-based real-time face detection and validation
- **Image Processing**: Advanced preprocessing with low-light enhancement
- **AJAX Integration**: Real-time communication between frontend and backend
- **Responsive Design**: Bootstrap-based responsive UI
- **Admin Interface**: Django admin for data management
- **Database Storage**: SQLite database with comprehensive data models

## 📋 Conversion from Streamlit

This Django application is a complete conversion of the original Streamlit `app.py` with the following improvements:

### Architecture Changes
| Component | Streamlit | Django |
|-----------|-----------|--------|
| **Frontend** | Streamlit widgets | HTML templates with Bootstrap CSS |
| **Backend** | Session state | Django views and models |
| **Database** | File system | SQLite with proper models |
| **Camera Integration** | Direct OpenCV | JavaScript MediaDevices API + AJAX |
| **Image Storage** | Local files | Django media files with database references |
| **User Interface** | Streamlit components | Professional web interface |

### Feature Mapping
- ✅ **StudentEnrollmentSystem class** → Django service (`services.py`)
- ✅ **Face detection logic** → Maintained with enhanced error handling
- ✅ **Interactive prompts** → Converted to dynamic web interface
- ✅ **Progress tracking** → Real-time updates via AJAX
- ✅ **Demo mode** → Enhanced file upload with validation
- ✅ **Image preprocessing** → Maintained with Django file handling

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Webcam (for live enrollment)
- Modern web browser (Chrome/Firefox recommended)

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Setup Script**
   ```bash
   python setup_django.py
   ```

3. **Create Superuser** (Optional)
   ```bash
   python manage.py createsuperuser
   ```

4. **Start Development Server**
   ```bash
   python manage.py runserver
   ```

5. **Access Application**
   - Main Application: http://127.0.0.1:8000/
   - Admin Interface: http://127.0.0.1:8000/admin/

### Manual Setup (Alternative)

```bash
# Make migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic

# Run server
python manage.py runserver
```

## 📱 Usage Guide

### 1. Dashboard
- View all enrolled students
- Track enrollment statistics
- Quick access to enrollment functions

### 2. Student Enrollment
1. Click "New Student" from dashboard
2. Fill in student information (Name, ID, Email, Phone)
3. Choose enrollment method:
   - **Live Camera**: Real-time camera capture
   - **Demo Mode**: Upload pre-captured images

### 3. Live Camera Enrollment
1. Grant camera permissions when prompted
2. Position face in camera view
3. Follow interactive prompts (10 different poses)
4. System automatically captures 100+ images
5. Real-time progress tracking and quality feedback

### 4. Demo Mode
1. Select multiple face images (JPG/PNG)
2. Upload and process images
3. System validates each image for face detection
4. View success/failure rates

## 🏗️ Project Structure

```
tz6_attendance/
├── main_app/                   # Main Django application
│   ├── models.py              # Student, EnrollmentImage, EnrollmentSession models
│   ├── views.py               # Web views and API endpoints
│   ├── services.py            # Face detection service (converted from Streamlit)
│   ├── admin.py               # Django admin configuration
│   ├── urls.py                # URL routing
│   └── templates/             # HTML templates
│       └── main_app/
│           ├── base.html      # Base template with Bootstrap
│           ├── dashboard.html # Main dashboard
│           ├── enrollment_form.html
│           ├── enrollment_capture.html  # Live camera interface
│           ├── demo_enrollment.html     # File upload interface
│           └── student_detail.html      # Student details view
├── tz6_attendance/            # Django project settings
│   ├── settings.py           # Configuration with media/static files
│   ├── urls.py               # Main URL configuration
│   └── wsgi.py               # WSGI configuration
├── static/                   # Static files (CSS, JS)
├── media/                    # Uploaded images
├── requirements.txt          # Updated dependencies
├── setup_django.py          # Setup script
└── manage.py                 # Django management script
```

## 🔧 Key Components

### Models (`models.py`)
- **Student**: Student information and enrollment status
- **EnrollmentImage**: Individual face images with metadata
- **EnrollmentSession**: Enrollment session tracking

### Services (`services.py`)
- **FaceDetectionService**: Core face detection and image processing
- Converted from original `StudentEnrollmentSystem` class
- Enhanced error handling and Django integration

### Views (`views.py`)
- **enrollment_dashboard**: Main dashboard view
- **enrollment_capture**: Live camera enrollment
- **demo_enrollment**: File upload enrollment
- **process_camera_frame**: AJAX endpoint for real-time processing

### Templates
- **Responsive Bootstrap design**
- **Real-time camera integration**
- **Progress tracking with visual indicators**
- **Professional admin interface**

## 🎯 Key Improvements Over Streamlit

1. **Professional UI**: Bootstrap-based responsive design
2. **Better Performance**: Optimized database queries and AJAX
3. **Scalability**: Proper web architecture with separation of concerns
4. **Data Management**: Comprehensive admin interface
5. **Security**: CSRF protection and proper file handling
6. **Deployment Ready**: Standard Django deployment options
7. **Extensibility**: Easy to add new features and integrations

## 🔐 Security Features

- CSRF protection on all forms
- File upload validation
- Input sanitization
- Secure media file handling
- Admin interface protection

## 📊 Database Schema

### Student Model
- student_id (Primary Key)
- full_name, email, phone
- enrollment_completed, total_images_captured
- enrollment_date, is_active

### EnrollmentImage Model
- Foreign Key to Student
- Image file with metadata
- Quality metrics (confidence, brightness, blur)
- Prompt index and sequence number

### EnrollmentSession Model
- Session tracking with statistics
- Success rates and duration
- Technical metadata (IP, User Agent)

## 🚀 Deployment

The application is ready for deployment using standard Django deployment methods:
- **Development**: `python manage.py runserver`
- **Production**: Gunicorn + Nginx
- **Cloud**: Heroku, AWS, Google Cloud
- **Docker**: Containerization ready

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with proper documentation
4. Test thoroughly
5. Submit pull request

## 📄 License

This project maintains the same license as the original Streamlit application.

## 🆘 Support

For issues, feature requests, or questions:
1. Check the Django logs for detailed error information
2. Verify camera permissions in browser
3. Ensure all dependencies are installed correctly
4. Review the setup steps in this README

---

**Migration Complete!** 🎉 Your Streamlit AI Attendance System is now a full-featured Django web application with enhanced functionality and professional UI.

## System Architecture Integration

### Step 1: Video Capture (Live Camera Feed)
- **Implementation**: `cv2.VideoCapture(0)` captures video at 30 FPS
- **Real-time Processing**: Each frame is immediately processed
- **Integration Point**: Feeds directly into Step 2 for face detection

### Step 2: Face Detection (Real-time Quality Control)
- **Algorithm**: Haar Cascade Classifier for fast detection
- **Quality Validation**: Multiple checks ensure good training data
  - Face size validation (minimum 50x50 pixels)
  - Position validation (face centered in frame)
  - Blur detection (Laplacian variance > 100)
  - Brightness validation (50-200 range)
  - Single face requirement (reject multiple faces)
- **Visual Feedback**: 
  - Green box = Good quality, ready to capture
  - Red box = Poor quality, needs adjustment
- **Integration Point**: Valid faces are passed to Step 3

### Step 3: Image Pre-processing (Real-time Enhancement)
- **Face Extraction**: Crop with 20% padding around detected face
- **Standardization**: Resize all faces to 128x128 pixels
- **Enhancement Pipeline**:
  1. Grayscale conversion
  2. Histogram equalization (improve contrast)
  3. Gaussian blur (reduce noise)
- **Data Storage**: Processed images saved with systematic naming
- **Integration Point**: Creates training dataset for recognition

## Key Integration Features

### 1. **Seamless Pipeline**
```
Live Video → Face Detection → Quality Check → Pre-processing → Storage
    ↓              ↓              ↓              ↓           ↓
  30 FPS      Real-time      Pass/Fail    Enhancement    Training Data
```

### 2. **Quality Control Loop**
- Only high-quality faces trigger image capture
- Automatic rejection of poor-quality frames
- User guidance for optimal positioning

### 3. **Guided Enrollment Process**
- **7 Different Poses**: Various angles and expressions
- **15 Images per Pose**: Comprehensive coverage
- **Total**: ~105 images per student
- **Real-time Feedback**: Immediate quality assessment

### 4. **Consistent Processing**
- Same pre-processing pipeline used in enrollment and recognition
- Ensures compatibility between training and testing phases

## Prototype Files

### 1. `enrollment_prototype.py`
**Main enrollment system with integrated Steps 1-3:**

**Key Classes:**
- `EnrollmentSystem`: Main class handling the enrollment process

**Key Methods:**
- `detect_and_validate_face()`: Step 2 implementation
- `preprocess_face_image()`: Step 3 implementation  
- `enroll_student()`: Complete enrollment pipeline

**Features:**
- Real-time face detection with quality validation
- Guided prompts for comprehensive data collection
- Automatic image capture when quality is good
- Live feedback to help users position correctly

### 2. `recognition_test.py`
**Testing system to validate enrollment results:**

**Key Classes:**
- `RecognitionTester`: Tests the trained model

**Key Methods:**
- `load_enrollment_data()`: Loads and trains on enrollment data
- `recognize_face()`: Performs face recognition
- `test_recognition()`: Live testing interface

**Features:**
- Uses same pre-processing as enrollment
- KNN classifier for face recognition
- Live recognition testing with confidence scores
- Results recording and analysis

## How to Use the Prototype

### Phase 1: Student Enrollment
```bash
/workspaces/tz6_attendance/.venv/bin/python enrollment_prototype.py
```

**Process:**
1. Enter student name and ID
2. Follow on-screen prompts for different poses
3. System automatically captures high-quality images
4. 105+ images saved per student

**Controls:**
- `q`: Quit enrollment
- `n`: Next prompt (after 5+ images)
- `s`: Skip current prompt

### Phase 2: Recognition Testing
```bash
/workspaces/tz6_attendance/.venv/bin/python recognition_test.py
```

**Process:**
1. System loads all enrollment data
2. Trains KNN classifier
3. Live camera feed with real-time recognition
4. Press 'r' to record recognition results

## Technical Details

### Face Detection Quality Metrics
- **Blur Threshold**: Laplacian variance > 100
- **Size Threshold**: Minimum 50x50 pixels
- **Position Tolerance**: Face center within 20% of frame center
- **Brightness Range**: 50-200 (0-255 scale)

### Image Pre-processing Pipeline
1. **Extract**: Face region + 20% padding
2. **Resize**: 128x128 pixels (standard size)
3. **Enhance**: Histogram equalization
4. **Denoise**: 3x3 Gaussian blur
5. **Save**: Grayscale JPEG format

### Recognition Model
- **Algorithm**: K-Nearest Neighbors (k=5)
- **Distance Metric**: Euclidean distance
- **Confidence Threshold**: 0.6 (60%)
- **Input**: Flattened 128x128 image (16,384 features)

## Data Structure
```
enrollment_data/
├── 001_John_Doe/
│   ├── pose_0_000.jpg  # Straight ahead
│   ├── pose_0_001.jpg
│   ├── ...
│   ├── pose_1_000.jpg  # Look up
│   └── ...
├── 002_Jane_Smith/
│   └── ...
```

## Performance Characteristics

### Enrollment Phase
- **Processing Speed**: 30 FPS real-time
- **Capture Rate**: ~3-5 images per second (when quality is good)
- **Time per Student**: 2-3 minutes for complete enrollment
- **Quality Assurance**: Only high-quality images stored

### Recognition Phase
- **Detection Speed**: Real-time (30 FPS)
- **Recognition Accuracy**: Depends on enrollment data quality
- **Response Time**: < 100ms per face
- **Confidence Scoring**: 0.0-1.0 range

## Next Steps for Enhancement

1. **Deep Learning Integration**: Replace KNN with CNN for better accuracy
2. **Face Landmark Detection**: More precise face alignment
3. **Data Augmentation**: Synthetic variations of enrollment data
4. **Database Integration**: Store student information in database
5. **Attendance Logging**: Complete attendance management system

This prototype successfully demonstrates the integration of video capture, face detection, and image pre-processing in a real-time enrollment system, providing a solid foundation for the complete attendance management system.
