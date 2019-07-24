# ---------------------- Importing flask stuff ----------------------

from flask import Flask, render_template, flash, request, url_for, redirect,  Response
from flask_sqlalchemy import SQLAlchemy
from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin, login_required, current_user
from flask_mail import Mail
from flask_security.forms import RegisterForm
from flask.ext.wtf import Form
from wtforms import TextField
from wtforms.validators import DataRequired

# ---------------------- Importing sqlalchemy stuff ----------------------

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, DateTime

# ---------------------- Importing cv stuff ----------------------

import cv2
import os
import numpy as np
import cv2.face
import time
import datetime

from crontab import CronTab

cron = CronTab(user=True)

# ---------------------- Importing custom classes ----------------------

from methods import FaceDetector
from methods import FaceRecognition

# ---------------------- Init app ----------------------

app = Flask(__name__, static_url_path='/static')

# ---------------------- App configuration ----------------------

app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'super-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@127.0.0.1/flaskapp'
app.config['SECURITY_REGISTERABLE'] = True
app.config['SECURITY_FLASH_MESSAGSE'] = True
app.config['SECURITY_PASSWORD_HASH'] = 'bcrypt'
app.config['SECURITY_LOGIN_USER_TEMPLATE'] = 'login.html'
app.config['SECURITY_REGISTER_USER_TEMPLATE'] = 'register.html'
app.config['SECURITY_PASSWORD_SALT'] = 'asdwer34'
app.config['SECURITY_CONFIRMABLE'] = False
app.config['SECURITY_SEND_REGISTER_EMAIL'] = False
app.config['SECURITY_POST_LOGIN_VIEW']= '/dashboard'

# ---------------------- Init db ----------------------
db = SQLAlchemy(app)

# ---------------------- Database models ----------------------

roles_users = db.Table('roles_users',
        db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
        db.Column('role_id', db.Integer(), db.ForeignKey('role.id')))

class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True)
    user_type = db.Column(db.String(255))
    password = db.Column(db.String(255))
    active = db.Column(db.Boolean())
    confirmed_at = db.Column(db.DateTime())
    roles = db.relationship('Role', secondary=roles_users,
                            backref=db.backref('users', lazy='dynamic'))


class ExtendedRegisterForm(RegisterForm):
    user_type = TextField('User Type', [DataRequired()])

    def validation(self):
        validation = Form.validate(self)
        if not validation:
            return False
        elif (validation=='student' or validation=='staff'):
            return True
        else:
            return False


# ---------------------- Initialize models ----------------------

user_datastore = SQLAlchemyUserDatastore(db, User, Role)
security = Security(app, user_datastore,
         register_form=ExtendedRegisterForm)
# security = Security(app, user_datastore)


#@app.before_first_request
#def create_user():
#    db.create_all()
#    user_datastore.create_user(email='admin', password='password', user_type='staff')
#    db.session.commit()


# ---------------------- Main views ----------------------

@app.route('/about')
@login_required
def aboutPage():
    return render_template('about.html')

@app.route('/dashboard')
@login_required
def mainDashboardPage():
    if(current_user.user_type=='staff'):
        return render_template('staff-home.html', current_user=current_user)
    else:
        return render_template('student-home.html', current_user=current_user)


@app.route('/')
# @login_required
def aboutDashPage():
    return render_template('dashboard.html')
    # if(current_user.user_type=='staff'):
    #     return render_template('staff-home.html', current_user=current_user)
    # else:
    #     return render_template('student-home.html', current_user=current_user)

# ---------------------- Staff students models and views ----------------------

engine = create_engine('mysql+pymysql://root:root@127.0.0.1/flaskapp')
Base = declarative_base()

class Students(Base):
    __tablename__ = 'students'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    year = db.Column(db.String(100))
    attendance = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey(User.id))
    #user = relationship('User', foreign_keys='Friend.user_id')

Session = sessionmaker(bind=engine)
Session.configure(bind=engine)
session = Session()
Base.metadata.create_all(engine)
session._model_changes = {}


@app.route('/staff/students', methods=['GET', 'POST'])
@login_required
def staffStudentPage():
    # if request.method == 'POST':
    # ed_user = Students(user_id=current_user.id, name="Krishna", year="BEIT", attendance="60")
    # session.add(ed_user)
    # session.commit()
    # else:
    if(current_user.user_type=='staff'):
        student_detail = session.query(User)
        return render_template('staff-students.html', student_detail=student_detail)
    else:
        return render_template('student-home.html')


# ---------------------- Initializing Face detector ----------------------

facedetection = FaceDetector(0)

# ---------------------- Face detector ----------------------
@app.route('/face_detection')
@login_required
def face_detection():
    return render_template('face_detection.html')

@app.route('/save_face_feed')
def save_face_feed():
    person_name = request.args.get('person_name')
    return Response(facedetection.start_detection(person_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------- Face Recognition ----------------------

facerecognition = FaceRecognition(0)

@app.route('/staff/attendance')
@login_required
def recognitionIndex():
    return render_template('staff-attendance.html')

@app.route('/recognition_feed')
def recognition_feed():
    return Response(facerecognition.start_recognition(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------- Attendance records ----------------------

class AttendUser(Base):
    __tablename__ = 'attendance_table'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255))
    date = Column(DateTime)
    present = Column(String(12))

    # if request.method == 'POST':
    # ed_user = Students(user_id=current_user.id, name="Krishna", year="BEIT", attendance="60")
    # session.add(ed_user)
    # session.commit()
    # else:

@app.route('/staff/attendance/records')
@login_required
def staffStudentRecordPage():
    if(current_user.user_type=='staff'):
        student_detail = session.query(AttendUser)
        return render_template('staff-student-record.html', student_detail=student_detail)


@app.route('/student-attendance')
@login_required
def singleStudentRecordPage():
    if(current_user.user_type=='student'):
        student_detail = session.query(AttendUser).filter(AttendUser.user_id == current_user.id)
        return render_template('student-attendance.html', student_detail=student_detail)



# ---------------------- Training the datamodel ----------------------

@app.route('/start-training')
@login_required
def startTraining():
    facerecognition.train_recognizer()
    return render_template('staff-attendance.html', message="Training successful")

@app.route('/timetable', methods=['GET', 'POST'])
@login_required
def timetableHandling():
    if request.method == 'POST':
        comma = request.form['name']
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])

        job = cron.new(command='/home/krishna/.virtualenvs/cv/bin/python /home/krishna/Project/Project/project-detect/silent_recognition.py',comment=comma)
        job.minute.on(minute)
        job.hour.on(hour)
        cron.write()
        # job =cron.find_comment('FaceRecognition')
        # cron.remove( job )

        # return render_template('staff-add-timetable.html', message=request.form['name'])
    else:
        return render_template('staff-add-timetable.html', message="")






# ---------------------- Error Handling ----------------------

@app.errorhandler(404)
def pageNotFound(e):
    return render_template('404.html')

@app.errorhandler(405)
def pageNotFound(e):
    return render_template('404.html')

@app.errorhandler(500)
def pageError(e):
    return render_template('404.html')

# ---------------------- Running app ----------------------

if __name__ == '__main__':
    app.run()
