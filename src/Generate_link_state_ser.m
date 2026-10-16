function [channel_state,out_ID] = Generate_link_state_ser(in_ID,loss_probability)
%
% The service:
%
%    [channel_state,out_ID] = Generate_link_state_ser(in_ID,loss_probability)
%
% returns a binary 0-1 r.v.: channel_state with Prob(channel_state == 0) ==
% oss_probability.
%T he service is implemented as a MATLAB function.
%
out_ID= in_ID;
%
X = max(0, min(1, rand(1)));
%R.v. X is evenly distributed over (0,1)
%
if(X < loss_probability)
   channel_state = 0;
else
   channel_state = 1;
end %End if
%channel_state is a binary r.v. with: Prob(channel_state == 0) ==
%loss_probability
%
end %End service code