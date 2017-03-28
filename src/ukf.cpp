#include "ukf.h"
#include "tools.h"
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
  // is_initialized_ is false before the process starts
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = false;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.01;

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

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Predicted sigma point matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 0; i < 2 * n_aug_; i++)
  {
    weights_(i + 1) = 0.5 / (lambda_ + n_aug_);
  }

  // Current NIS for radar
  NIS_radar_ = 0.0;

  // Current NIS for laser
  NIS_laser_ = 0.0;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  Initialize the first measurement, predict and update
  */
  // Initialize the first data
  if (!is_initialized_) {// then it's the first measurement
    // set time stamp
    time_us_ = meas_package.timestamp_;

    // initialize state vector
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { // Radar measurement
      // Convert radar from polar to cartesian coordinates and initialize state.
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];

      x_ << rho * cos(phi), rho * sin(phi), rho_dot, 0.05, 0.05; // v approximated as rho_dot
    }
    else { // Laser measurement
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0.05, 0.05;
    }

    // initialize state covariance matrix
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1000, 0, 0,
          0, 0, 0, 10, 0,
          0, 0, 0, 0, 10;

    // done initializing, set is_initialized
    is_initialized_ = true;
  }

  // Calculate delta_t
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;

  // Predict
  Prediction(delta_t);

  // Update
  if ((meas_package.sensor_type_ == MeasurementPackage::LASER) & (use_laser_)) {
    UpdateLidar(meas_package);
  }
  else if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) & (use_radar_)) {
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Estimate the object's location. Modify the state vector, x_. Predict sigma
  points, the state, and the state covariance matrix.
  */
  // Generate sigma points
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
  MatrixXd A = P_.llt().matrixL();
  Xsig.col(0) = x_;
  for (int i = 0; i < n_x_; i++)
  {
    Xsig.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig.col(i + n_x_ + 1) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }

  // Augment sigma points
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_,n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1,n_x_ + 1) = std_yawdd_ * std_yawdd_;

  MatrixXd sqrtMatrix = P_aug.llt().matrixL();
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * sqrtMatrix.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * sqrtMatrix.col(i);
  }

  // Predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Extract x_k
    VectorXd x = Xsig_aug.col(i);
    VectorXd x_k = x.head(n_x_);

    // Calculate the deterministic term
    VectorXd d_term = VectorXd(n_x_);
    if (fabs(x(4)) > 0.001) // when yaw rate is NOT zero
    {
      d_term(0) = x(2)/x(4) * (sin(x(3) + x(4) * delta_t) - sin(x(3)));
      d_term(1) = x(2)/x(4) * (-cos(x(3) + x(4) * delta_t) + cos(x(3)));
    }
    else // when yaw rate is zero
    {
      d_term(0) = x(2) * cos(x(3)) * delta_t;
      d_term(1) = x(2) * sin(x(3)) * delta_t;
    }
    d_term(2) = 0;
    d_term(3) = x(4) * delta_t;
    d_term(4) =0;

    // Calculate the stachastic term s_term
    VectorXd s_term = VectorXd(5);
    s_term(0) = 0.5 * delta_t * delta_t * cos(x(3)) * x(5);
    s_term(1) = 0.5 * delta_t * delta_t * sin(x(3)) * x(5);
    s_term(2) = delta_t * x(5);
    s_term(3) = 0.5 * delta_t * delta_t * x(6);
    s_term(4) = delta_t * x(6);

    // Add three terms together
    Xsig_pred_.col(i) = x_k + d_term + s_term;
  }

  // Predict state vector
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    x = x + weights_(i) * Xsig_pred_.col(i);
  }
  x_ = x;

  // Predict state covariance
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P = P + weights_(i) * x_diff * x_diff.transpose();
  }
  P_ = P;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Use lidar data to update the belief about the object's position;
  Calculate lidar NIS.
  */
  // Extract measurement data
  VectorXd z = meas_package.raw_measurements_;

  // Define measurement space dimension
  int n_z = 2;

  // Map sigma points to measurement space
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, 2 * n_aug_ + 1);

  // Predict measurement mean
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Predict measurement covariance
  MatrixXd S = MatrixXd(n_z, n_z);
  MatrixXd R = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;
  S = S + R;

  // Update measurement
    // Calculate cross-correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

    // Calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

    // Update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Calculate NIS
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  Use radar data to update the belief about the object's position;
  Calculate radar NIS.
  */
  // Extract measurement data
  VectorXd z = meas_package.raw_measurements_;

  // Define measurement space dimension
  int n_z = 3;

  // Map sigma points to measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2* n_aug_ + 1; i++) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y); //rho
    Zsig(1, i) = atan2(p_y, p_x);             //phi
    Zsig(2, i) = (p_x * cos(yaw) + p_y * sin(yaw)) * v / Zsig(0, i); //rho_dot
  }

  // Predict measurement mean
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
    while (z_pred(1)> M_PI) z_pred(1)-=2.*M_PI;
    while (z_pred(1)<-M_PI) z_pred(1)+=2.*M_PI;
  }

  // Predict measurement covariance
  MatrixXd S = MatrixXd(n_z, n_z);
  MatrixXd R = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;
  S = S + R;

  // Update measurement
    // Calculate cross-correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

    // Calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

    // Update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Calculate NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

}


