function[AbsorbCoef]=CoefCalculate(F,T)
pres=1;
relh=50;
freq_hum=F;
temp=T+273;
C_humid=4.6151-6.8346*((273.15/temp)^1.261);
hum=relh*(10^C_humid)*pres;	
tempr=temp/293.15; 
frO=pres*(24+4.04e4*hum*(0.02+hum)/(0.391+hum));
frN=pres*(tempr^-0.5)*(9+280*hum*exp(-4.17*((tempr^-1/3)-1)));
alpha=8.686*freq_hum*freq_hum*(1.84e-11*(1/pres)*sqrt(tempr)...
+(tempr^-2.5)*(0.01275*(exp(-2239.1/temp)*1/(frO+freq_hum*freq_hum/frO))...
+0.1068*(exp(-3352/temp)*1/(frN+freq_hum*freq_hum/frN))));
db_humi=alpha;
db_humi =round(1000*db_humi)/1000;
AbsorbCoef=db_humi;
end

