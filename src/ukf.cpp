#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  use_laser_ = use_radar_ = false;
  time_us_ = 0;

  weights_ = VectorXd::Zero(2 * n_aug_ + 1);
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2*n_aug_ + 1);

  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  std::cout << "Weights: " <<weights_ << std::endl;
  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_) * 0.1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  std::cout << "Incoming measurement: " << meas_package.raw_measurements_ << std::endl;
  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) return;

  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;    //dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
      x_[0] = meas_package.raw_measurements_[0];
      x_[1] = meas_package.raw_measurements_[1];
    } else {
      x_[0] = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);
      x_[1] = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
    }
    is_initialized_ = true;
    return;
  }
  MatrixXd Xsig_aug = GenerateAugSigmaPoints();
  std::cout << "Pred: " << std::endl << Xsig_aug << std::endl;
  Prediction(Xsig_aug, dt);
  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
    UpdateLidar(meas_package);
  } else {
    UpdateRadar(meas_package);
  }


/**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(MatrixXd sig, double delta_t) {
  int sig_n = 2 * n_aug_ + 1;
  VectorXd sig_col = VectorXd(n_aug_);
  for (int i = 0; i < sig_n ; i++) {
    sig_col = sig.col(i);
    double yawdd = sig_col(4);
    VectorXd noise = VectorXd(n_x_);
    noise = VectorXd::Zero(n_x_);
    VectorXd iter_col = VectorXd::Zero(n_x_);
    noise << 0.5 * delta_t * delta_t * cos (sig_col[3]) * sig_col[5],
      0.5 * delta_t * delta_t * sin (sig_col[3]) * sig_col[5],
      delta_t * sig_col[5],
      0.5 * delta_t * delta_t * sig_col[6],
      delta_t * sig_col[6];
    if (yawdd == 0) {
      iter_col[0] = sig_col[2] * cos (sig_col[3]) * delta_t;
      iter_col[1] = sig_col[2] * sin (sig_col[3]) * delta_t;
    } else {
      iter_col[0] = sig_col[2] / sig_col[4] * (sin(sig_col[3] + sig_col[4] * delta_t) - sin(sig_col[3]));
      iter_col[1] = sig_col[2] / sig_col[4] * (-cos(sig_col[3] + sig_col[4] * delta_t) + cos(sig_col[3]));
      iter_col[3] = delta_t * sig_col[4];
    }
    iter_col += noise;
    iter_col += sig_col.head(5);
    Xsig_pred_.col(i) = iter_col;
  }

  std::cout << "XsigPred-PostPrediction: " << std::endl << Xsig_pred_ << std::endl;
  std::cout << "X is " << x_ <<std::endl;
  x_.fill(0);
  P_.fill(0);
  x_ = weights_[0] * Xsig_pred_.col(0);
  for (int i = 1; i < sig_n ; i++ ) {
    x_ += weights_[i] * Xsig_pred_.col(i);
  }
  std::cout << "Xpred: " << std::endl << x_ << std::endl;

  for (int i = 0; i < sig_n; i++ ) {
    VectorXd x_diff = (Xsig_pred_.col(i) - x_);
    NormAng(&(x_diff[3]));
    P_ += weights_[i] * ( x_diff * (Xsig_pred_.col(i) - x_).transpose());
  }
  std::cout << "P: " << std::endl << P_ << std::endl;

  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  int n_z = 2;
  MatrixXd Zsig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z,n_z);

/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  for (int i=0; i < 2*n_aug_ +1; i++) {
    VectorXd iter_col  = Xsig_pred_.col(i);
    Zsig.col(i)[0] =  iter_col[0];
    Zsig.col(i)[1] =  iter_col[1];
    z_pred += weights_[i] * Zsig.col(i);

  }

  for (int i = 0 ; i < 2*n_aug_ + 1; i++) {
    S += weights_[i] * ((Zsig.col(i) - z_pred) * (Zsig.col(i) - z_pred).transpose());
  }
  S(0,0) += std_laspx_ * std_laspx_;
  S(1,1) += std_laspy_ * std_laspy_;

  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

/*******************************************************************************
 * Student part begin
 ******************************************************************************/
  for (int i = 0; i < n_aug_ *2 + 1; i++) {
    VectorXd x_diff = ( Xsig_pred_.col(i) - x_ );
    NormAng(&x_diff[3]);
    Tc += weights_[i] * ( x_diff  * (Zsig.col(i) - z_pred).transpose());
  }
  MatrixXd K = MatrixXd(n_x_, n_z);
  K = Tc * S.inverse();

  x_ = x_ + K * (meas_package.raw_measurements_ - z_pred);
  //calculate cross correlation matrix
  //calculate Kalman gain K;
  //update state mean and covariance matrix
  P_ = P_ - K * S * K.transpose();

  std::cout << "Laser Update, x is : " << x_  <<std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  int n_z = 3;
  MatrixXd Zsig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z,n_z);

/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  for (int i=0; i < 2*n_aug_ +1; i++) {
    VectorXd iter_col  = Xsig_pred_.col(i);
    Zsig.col(i)[0] =  sqrt(iter_col[0] * iter_col[0] + iter_col[1] * iter_col[1]);
    Zsig.col(i)[1] = atan( iter_col[1] / iter_col[0] );
    Zsig.col(i)[2] = (iter_col[0] * cos(iter_col[3]) * iter_col[2] + iter_col[1] * sin(iter_col[3]) * iter_col[2]) / Zsig.col(i)[0];
    z_pred += weights_[i] * Zsig.col(i);

  }

  std::cout << "zPred: " << std::endl << z_pred << std::endl;
  for (int i = 0 ; i < 2*n_aug_ + 1; i++) {
    VectorXd zdiff = (Zsig.col(i) - z_pred);
    NormAng(&zdiff[1]);
    S += weights_[i] * (zdiff * (zdiff).transpose());
  }
  S(0,0) += std_radr_ * std_radr_;
  S(1,1) += std_radphi_ * std_radphi_;
  S(2,2) += std_radrd_ * std_radrd_;


  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);


/*******************************************************************************
 * Student part begin
 ******************************************************************************/
  for (int i = 0; i < n_aug_ *2 + 1; i++) {
    VectorXd x_diff = ( Xsig_pred_.col(i) - x_ );
    NormAng(&x_diff[3]);
    VectorXd zdiff = (Zsig.col(i) - z_pred);
    NormAng(&zdiff[1]);
    Tc += weights_[i] * (x_diff  * zdiff.transpose());
  }
  MatrixXd K = MatrixXd(n_x_, n_z);
  K = Tc * S.inverse();


  VectorXd zdiff = (meas_package.raw_measurements_ - z_pred);
  NormAng(&zdiff[1]);

  x_ = x_ + K * zdiff;
  //calculate cross correlation matrix
  //calculate Kalman gain K;
  //update state mean and covariance matrix
  P_ = P_ - K * S * K.transpose();
  std::cout << "Radar Update, x is : " << x_  <<std::endl;


}


MatrixXd UKF::GenerateAugSigmaPoints() {

  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_ * std_a_, 0,
    0, std_yawdd_ * std_yawdd_;

//create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

/*******************************************************************************
 * Student part begin
 ******************************************************************************/
  x_aug.head(5) = x_;
  x_aug[5] = 0;
  x_aug[6] = 0;
  P_aug.topLeftCorner(5,5) = P_;
  P_aug.bottomRightCorner(2,2) = Q;

  P_aug *= lambda_ + n_aug_;

  //calculate square root of P
  MatrixXd A = P_aug.llt().matrixL();

  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_ ; i++) {
    Xsig_aug.col(i+1) = x_aug + A.col(i);
    Xsig_aug.col(i+n_aug_+1) = x_aug - A.col(i);
  }
  return Xsig_aug;
}

void UKF::NormAng(double *ang) {
  while (*ang > M_PI) *ang -= 2. * M_PI;
  while (*ang < -M_PI) *ang += 2. * M_PI;
}