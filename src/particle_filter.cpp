/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	 
	num_particles = 100;
	double init_weight = 1.0f;
	
	//Gassian distribution around specified operating point.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	for (int i = 0; i < num_particles; i++) {
		double sample_x, sample_y, sample_theta;
				
        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);
		
		Particle particle;
		particle .id = i;
		particle.x = sample_x;
		particle.y = sample_y;
		particle.theta = sample_theta;
		particle.weight = init_weight;		
		
		particles.push_back(particle);
		weights.push_back(init_weight);
	}
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;
	
	//Gaussian Noise with zero mean
    normal_distribution<double> noise_x(0,std_pos[0]);
    normal_distribution<double> noise_y(0,std_pos[1]);
    normal_distribution<double> noise_theta(0,std_pos[2]);
	
	for (int i = 0; i < num_particles; i++){
		double i_theta = particles[i].theta;
		
		if (fabs(yaw_rate)> 0.001){
			particles[i].x += (velocity/yaw_rate)*(sin(i_theta +(yaw_rate*delta_t))- sin(i_theta));
			particles[i].y += (velocity/yaw_rate)*(cos(i_theta)-cos(i_theta +(yaw_rate*delta_t)));
			particles[i].theta += (yaw_rate*delta_t);
		} else {
			particles[i].x += velocity*delta_t*cos(i_theta);
			particles[i].y += velocity*delta_t*sin(i_theta);
		}
		
		particles[i].x += noise_x(gen);
		particles[i].y += noise_y(gen);
		particles[i].theta += noise_theta(gen);
		
	}

}

/**
 * dataAssociation Finds which observations correspond to which landmarks (likely by using
*   a nearest-neighbors data association).
* @param predicted Vector of predicted landmark observations
* @param observations Vector of landmark observations
*/
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for (int i = 0; i < observations.size(); i++){
		
		LandmarkObs i_obs = observations[i];
		double min_dist = 1000000; // a lot bigger than sensor_range
		int land_id = -1;
		
		for (int j = 0; j < predicted.size(); j++){
			LandmarkObs j_pred = predicted[j];
			
			double j_dist = dist(i_obs.x,i_obs.y,j_pred.x,j_pred.y);
			
			if (j_dist < min_dist){
				min_dist = j_dist;
				land_id = j_pred.id;
			}
		}
		
		observations[i].id = land_id;
	}

}
/**
 * updateWeights Updates the weights for each particle based on the likelihood of the 
 *   observed measurements. 
 * @param sensor_range Range [m] of sensor
 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
 * @param observations Vector of landmark observations
 * @param map Map class containing map landmarks
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	for (int i = 0; i < num_particles; i++){
		
		double i_px = particles[i].x;
		double i_py = particles[i].y;
		double i_ptheta = particles[i].theta;
		
		particles[i].weight = 1.0f;
		
		//sensor raange to select the predicted landmarks 
		vector<LandmarkObs> predicted;
		
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
			int j_mapid = map_landmarks.landmark_list[j].id_i;
			double j_mapx = map_landmarks.landmark_list[j].x_f;
			double j_mapy = map_landmarks.landmark_list[j].y_f;
			
			double map_dist = dist(i_px,i_py,j_mapx,j_mapy);
			
			if (map_dist<sensor_range){
				 
				 LandmarkObs mark;
				 mark.id = j_mapid;
				 mark.x = j_mapx;
				 mark.y = j_mapy;
				
				predicted.push_back(mark);
			}
		
		}
		
		//Transform Measurements to their global frame
		vector<LandmarkObs> observations_map ;
		
		for (int k = 0; k < observations.size(); k++){
			
			double k_obsx = observations[k].x;
			double k_obsy = observations[k].y;
			
			LandmarkObs obs_map;
			
			obs_map.id = observations[k].id; 
			obs_map.x = i_px + (cos(i_ptheta)*k_obsx) -(sin(i_ptheta)*k_obsy);
			obs_map.y = i_py + (sin(i_ptheta)*k_obsx) +(cos(i_ptheta)*k_obsy);
			
			observations_map.push_back(obs_map);
			
		}
	    
		//Perform Landmark association
		dataAssociation(predicted,observations_map);
		
		for (int l = 0; l < observations_map.size(); l++){
			int l_obsid = observations_map[l].id;
			double l_obsx = observations_map[l].x;
			double l_obsy = observations_map[l].y;
			double l_predx, l_predy;
			
			for (int m = 0; m < predicted.size(); m++){
				
				if (predicted[m].id == l_obsid){
					
					l_predx = predicted[m].x;
					l_predy = predicted[m].y;
					
				}
				
			}
			double l_gnorm = (1/(2*M_PI*std_landmark[0]*std_landmark[1]));
			double l_exponent = (pow((l_predx - l_obsx),2))/(2 * pow( std_landmark[0],2)) + (pow((l_predy - l_obsy),2))/(2 * pow( std_landmark[1],2));
			
			particles[i].weight *= l_gnorm*exp(-l_exponent);
			
		}
		
		weights[i] = particles[i].weight;
		
	}
	
}
/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> int_particles;
	
	discrete_distribution<int> dist_particle(weights.begin(),weights.end());
	
	for (int i = 0; i < num_particles; i++){
		int j = dist_particle(gen);
		
		Particle j_particle;
		j_particle.id = i;
		j_particle.x = particles[j].x;
		j_particle.y = particles[j].y;
		j_particle.theta = particles[j].theta;
		j_particle.weight = particles[j].weight;
		
		int_particles.push_back(j_particle);
	}
	
	particles = int_particles;
	
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
