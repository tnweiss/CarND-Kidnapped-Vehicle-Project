/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::numeric_limits;
using std::uniform_real_distribution;
using std::uniform_int_distribution;


static std::default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * DONE: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * DONE: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  // completed with help from this lesson https://classroom.udacity.com/nanodegrees/nd013/parts/b9040951-b43f-4dd3-8b16-76e7b52f4d9d/modules/85ece059-1351-4599-bb2c-0095d6534c8c/lessons/e3981fd5-8266-43be-a497-a862af9187d4/concepts/226a0ca8-f66a-42d5-ac96-e37019fd6f15
  num_particles = 100;  // TODO: Set the number of particles
  
  // Create normal distributions for x, y, and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i = 0; i < num_particles; i++) {
    // initialize the particle object
    Particle particle;
    
    // set the id to the index in the array
    particle.id = i;
    
    // generate random x, y, and theta samples
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    
    // all particles have equal weight
    particle.weight = 1.0;

    // add the particle to the list of particles
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * DONE: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  // Create normal distributions for x, y, and theta
  normal_distribution<double> nml_x(0, std_pos[0]);
  normal_distribution<double> nml_y(0, std_pos[1]);
  normal_distribution<double> nml_theta(0, std_pos[2]);
  
  for (int i = 0; i < num_particles; i++) {

    // if the yaw rate is not significant enough (effectively 0) then we don't factor that into the new theta, x and y
    if (fabs(yaw_rate) < 0.0000001) {  
      // multiply change in time (s) by veloctiy (m/s) times sin/cos to get the change in position with respect to that dimension
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      // factor in theta to the change in x and y https://classroom.udacity.com/nanodegrees/nd013/parts/b9040951-b43f-4dd3-8b16-76e7b52f4d9d/modules/85ece059-1351-4599-bb2c-0095d6534c8c/lessons/e3981fd5-8266-43be-a497-a862af9187d4/concepts/56d08bf5-8668-42e7-a718-1ef40d444259
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      // if the vehicle has turned we need to calculate the new theta to get the new angle 
      particles[i].theta += yaw_rate * delta_t;
    }

    // add noise to the predictions
    particles[i].x += nml_x(gen);
    particles[i].y += nml_y(gen);
    particles[i].theta += nml_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * DONE: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  double minD;
  double distance;
  
  // loop through each observation and find its nearest point
  for (unsigned int i = 0; i < observations.size(); i++) {
    LandmarkObs o = observations[i];
    
    // start off with the biggest number possible so we don't initialize something less than the actual min distance
    minD = numeric_limits<double>::max();
    
    // calculate the distance from the observation to every particle
    for (unsigned j = 0; j < predicted.size(); j++ ) {
      LandmarkObs p = predicted[j];

      distance = dist(o.x, o.y, p.x, p.y);

      // If the "distance" is less than min, stored the id and update min.
      if ( distance < minD ) {
        minD = distance;
        observations[i].id = p.id;
      }
    }
    
  }
  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * DONE: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double sr_sq = sensor_range * sensor_range;

  for (int i = 0; i < num_particles; i++) {
    // pull x, y, and theta from the current particle
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
    
    vector<LandmarkObs> lndmkInRange;
    
    // get all the landmarks within range of our sensor
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      float lx = map_landmarks.landmark_list[j].x_f;
      float ly = map_landmarks.landmark_list[j].y_f;
      int lid = map_landmarks.landmark_list[j].id_i;
      
      double dx = lx - x;
      double dy = ly - y;
      double ldist = (dx * dx) + (dy * dy);
      
      // if the landmark is in range of the sensor add it to the list of landmarks for our sensor
      if ( ldist <= sr_sq ) {
        lndmkInRange.push_back(LandmarkObs{ lid, lx, ly });
      }
    }
    
    // create an array of observations and map coordinates relative to the vehicle to map coordinates
    vector<LandmarkObs> tobs;
    for (unsigned int j = 0; j < observations.size(); j++) {
      LandmarkObs lndmk;
      double tobs_x = (cos(theta) * observations[j].x) - (sin(theta) * observations[j].y) + x;
      double tobs_y = (sin(theta) * observations[j].x) + (cos(theta) * observations[j].y) + y;
      lndmk.x = tobs_x;
      lndmk.y = tobs_y;
      lndmk.id = observations[j].id;
      tobs.push_back(lndmk);
    }
    
    // associate predicitons with observations
    dataAssociation(lndmkInRange, tobs);
    
    // reset 
    particles[i].weight = 1.0;
    weights[i] = 1.0;
    
    
    for (unsigned int j = 0; j < tobs.size(); j++) {
      
      // placeholders for observation and associated prediction coordinates
      double obs_x = tobs[j].x;
      double obs_y = tobs[j].y;
      double pr_x;
      double pr_y;

      // get the id of the predicted particle
      int associated_prediction = tobs[j].id;

      // get the x,y coordinates of the predicted particle
      for (unsigned int m = 0; m < lndmkInRange.size(); m++) {
        if (lndmkInRange[m].id == associated_prediction) {
          pr_x = lndmkInRange[m].x;
          pr_y = lndmkInRange[m].y;
          break;
        }
      }

      // multivariate Gaussian formula
      // https://knowledge.udacity.com/questions/217113
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double weight = ( 1 / (2 * M_PI * std_x * std_y)) * exp( -( pow(obs_x - pr_x, 2) / ( 2 * pow(std_x, 2)) + (pow(obs_y - pr_y, 2) / ( 2 * pow(std_y, 2)))));

      // product of this obersvation weight with total observations weight      
	  particles[i].weight *= weight;
    }
    
    weights[i] = particles[i].weight;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::discrete_distribution<int> dd(weights.begin(), weights.end());
  
  vector<Particle> newP;
  for(int i = 0; i < num_particles; i++) {
      int newParticleIndex = dd(gen);
      newP.push_back(particles[newParticleIndex]);
  }
  
  particles = newP;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  //Clear the previous associations
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}