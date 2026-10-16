function [out_channel_gain,out_ID,out_d] = Generate_link_gain_ser(in_ID, in_d_max,...
    in_d_min, in_path_loss_exponent, in_carrier_f, in_TX_by_RX_antenna_gains_product, in_d, in_UAV_H, in_FL_environment_flag, in_B, in_C, in_eta_LoS, in_eta_NLoS)
%
% The service:
%
% [out_channel_gain,out_ID, out_d] = Generate_link_gain_ser(in_ID, in_d_max,
%    in_d_min, in_path_loss_exponent, in_carrier_f, in_TX_by_RX_antenna_gains_product, in_d, in_UAV_H, in_FL_environment_flag, in_B, in_C, in_eta_LoS, in_eta_NLoS)
%
% generate a power channel-gain that accounts for (see the input parameters):
% (i) the (possibly, random) coworker-to-orchestrator distance (in (m));
% (ii) the product of the transmit and receive antenna gains (in linear scale); 
% (iii) the (dimensionless, and, possibly, random) power-fading coefficient of unit-valued expectation;
% (iv) the (dimensionless) path loss exponent; 
% (v) the carrier frequency in Hertz;
% and, 
% (vi) the parameters: B, C, eta_LoS and eta_NLos of the emulated logistic 
% path loss model in the case of drone-supported FL environment. 
%
% All these input parameters refer to a (unicast) coworker-to-orchestrator up-link.
% The (broadcast) orchestrator-to-coworker down-link is assumed to be ideal, e.g., fading-free, 
% packet loss-free, noise-free, delay-free and attenuation-free.
%
% By default,
% 1) the coworker-to-orchestrator distance is generated as a Markovian random
% walk over the allowed interval [in_d_min, in_d_max], whereas
% 2)the generated fading is of Rayleigh type, with unit-valued second-order
% moment.
%
out_ID = in_ID;
%
%-----Generate a random coworker-to-orchestrator distance falling in the---
%--------specified interval [in_d_min, in_d_max]--------------------------
gamma = 0.8; 
%Forgetting factor of the generated Markovian random walk.
%It must fall into the closed interval: 0 <= gamma <= 1.0
%
if (in_d_min <= in_d_max)
    d_sample = max(in_d_min, min(in_d_max, (gamma*in_d)+((1-gamma)*(in_d_min + ((in_d_max - in_d_min)*rand(1))))));
    %d is a Markovian r.v. distributed over [in_d_min, in_d_max]
    %This part of the code can be customized by the user on the basis of  
    %the emulated coworker mobility model.
else
    error('FedE error from Generate_link_gain_ser: d_min cannnot be larger than d_max');
end
%
c = 3*(10^8); %Light speed in (m/sec)
%
switch in_FL_environment_flag 
    case 0 %The emulated FL environment is of terrestrial type
         power_fading_coefficient = 0.5*(((randn(1))^2)+((randn(1))^2));
         %Generate an exponentially distributed non-negative r.v. with unit-valued
         %expectation. It corresponds to Rayleigh fading. 
         %This part can be customized by the user in agreement with the desired
         %(e.g., Rayleigh, Rice, Nakagami, Probabilistic etc) fading distribution.
         %In the case of deterministic (e.g., fading-free) coworker-to-orchestrator
         %terrestrial channel, the following setting would take place: power_fading_coefficient = 1.
         out_channel_gain = (in_TX_by_RX_antenna_gains_product)*(power_fading_coefficient)...
                            *((c/(4*pi*in_carrier_f))^2)*((1/(1+d_sample))^in_path_loss_exponent);

    case 1 %The emulated FL environment is drone-supported
         in_eta_LoS_lin = 10^(0.1*in_eta_LoS);   %Decibel-to-linear scale
         in_eta_NLoS_lin = 10^(0.1*in_eta_NLoS); %translation
         theta_sample = max(0, min(90, asind(in_UAV_H/d_sample))); %Generated the elevation angle (degree)
         LoS_probability_sample = 1/(1+(in_C*exp(-in_B*(theta_sample - in_C)))); 
         %LoS probability sample evaluated in accordance to the logistic probability model  
         path_loss_sample = (1/in_TX_by_RX_antenna_gains_product)*(((4*pi*in_carrier_f)/c)^2)*((1+d_sample)^in_path_loss_exponent)*(in_eta_NLoS_lin - ((in_eta_NLoS_lin - in_eta_LoS_lin)*LoS_probability_sample));
         %Path loss-sample evaluated in accordance to the adopted probabilistic
         %path-loss model         
         out_channel_gain = 1/path_loss_sample;         
end %End switch

%
out_d = d_sample; %Currently generated coworker-orchestrator distance (in (m))
%
end %End service code 