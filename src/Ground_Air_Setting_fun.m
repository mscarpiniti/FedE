function  [d_max, d_min, av_ground_to_air_per_link_packet_loss_probability] = Ground_Air_Setting_fun(COV_R, UAV_R, UAV_H, B, C, sample_number)
%
% The function: 
%   [d_max, d_min, av_ground_to_air_per_link_packet_loss_probability] = Ground_Air_Setting_fun(COV_R, UAV_R, UAV_H, B, C, sample_number)
% computes and returns the system parameters of the emulated drone-supported
% Federated Learning environment. The logistic model is used as
% probabilistic model to numerically profile the average ground-to-air per-link packet
% loss probability under the environment featured by the input parameters B and C. 
%
if((UAV_R < 0)||(COV_R <= 0)|| (UAV_R > COV_R) || (UAV_H <= 0) || (B <= 0) || (C <= 0))
  error('FedE error: at least one of the input parameters passed to Ground_Air_Setting_fun is not allowed. FedE aborts');
end %End if
%
d_min = UAV_H;
%Minimum coworker-drone distance (m)

d_max = +sqrt((UAV_H^2)+((COV_R+UAV_R)^2));
%Maximum coworker-drone distance (m)
%
acc = 0; 

for j=1:sample_number
    d_sample = d_min + (rand(1)*(d_max - d_min)); 
    %Generate a random coworker-drone distance sample

    theta_sample = max(0, min(90, asind(UAV_H/d_sample)));
    %Elevation angle (in (deg)) corresponding to the generated d_sample

    packet_LoS_probability_sample = 1/(1+(C*exp(-B*(theta_sample - C))));
    packet_loss_probability_sample = 1 - packet_LoS_probability_sample;
    %The logistic model with parameters B and C is used for evaluating the current NLoS_probability_sample == packet_loss_probability_sample 
    acc = acc + packet_loss_probability_sample;
end %End for

av_ground_to_air_per_link_packet_loss_probability = (acc/sample_number);

if((av_ground_to_air_per_link_packet_loss_probability >= 0) && (av_ground_to_air_per_link_packet_loss_probability <= 1))
else
   error('FedE error: the packet loss probability value profiled by Ground_Air_Setting_fun is out of the interval [0 , 1]. FedE aborts');
end %End if
%
end %End function