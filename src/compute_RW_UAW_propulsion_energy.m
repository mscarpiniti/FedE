function [propulsion_energy] = compute_RW_UAW_propulsion_energy(gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, UAV_R, UAV_omega, training_wall_clock_time)
% The function:
%
% [propulsion_energy] = compute_RW_UAW_propulsion_energy(gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, UAV_R, UAV_omega, training_wall_clock_time)
%
% evaluates the propulsion energy (in (Joule)) of the emulated drone that
% hosts the orchestrator.
% By default, the simulated drone is Rotary-Wing (RW) 
% and follows a circular trajectory of (possibly, vanishing)
% radius UAV_R (m) at constant angular speed UAV_omega (rad/sec) and
% altitude.
% The vanishing UAV_R value corresponds to the case in which the simulated 
% RW drone stays aloft in the air at the center of the ground coverage area.  

UAV_speed = UAV_omega*UAV_R;
%Linear speed (in (m/sec)) of the RW-UAW
UAV_cent_acc = (UAV_omega^2)*UAV_R;
%Centripetal acceleration (in (m/(sec^2))) of the simulated RW drone
%
propulsion_power = (gamma_1*(1+(gamma_2*(UAV_speed^2)))) +...
    (gamma_3*(UAV_speed^3)) +...
    (gamma_4*sqrt((1+((UAV_cent_acc^2)/(9.81^2)))*(sqrt(1+((UAV_cent_acc^2)/(9.81^2))+((UAV_speed^4)/(gamma_5^2)))-((UAV_speed ^2)/gamma_5))));
%Propulsion power (in (Watt)) of the simulated RW drone
%
fprintf('FedE message: the drone propulsion power is %g (Watt)\n', round(propulsion_power,2));
propulsion_energy = propulsion_power*training_wall_clock_time;
%Returned propulsion energy (in (Joule)) of the simulated RW drone
end % End function