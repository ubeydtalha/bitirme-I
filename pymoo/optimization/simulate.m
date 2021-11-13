function [bw,gain,pm,power,area, Vth_, Vgs_, Vds_] = simulate(N,p,population,Cload,transistor_count)
format long
global  Rac1
Vth_ = zeros(N, 6);
for i=1:N
    fid = fopen('designparam.cir');
    Rac = textscan(fid, '%s %s %s %n','headerLines', 1);
    fwrite= '.PARAM';
    dlmwrite('designparam.cir', fwrite, 'delimiter','');
    for j=1:p
        fwrite1= ['+ ' char(Rac{2}(j)) ' = ' num2str(population(i,j))];
        dlmwrite('designparam.cir', fwrite1, '-append','delimiter','');
    end  
    fclose(fid);    
    system('start/min/wait C:\synopsys\Hspice_A-2008.03\BIN\hspicerf.exe amp.sp -o amp');
    %check operating regions of the transistors
    flag = zeros(1,N);
        
        fid3 = fopen('amp.dp0');
        Rac3 = textscan(fid3, '%s','headerLines', 3);
        fclose(fid3);

        % t is the number of transistors!
        cut_off(1,N)=0;
        triode(1,N)=0;
        index = 0;
        for t = 1:6
            transistor_name = sprintf('M%d',t);
            for k=1:size(Rac3{1})
                if (ismember(transistor_name,Rac3{1}(k)) == 1)
                index = k;
                break;
                end
            end
            Vgs(1,t) = abs(str2double(Rac3{1}(index + 68)));
            Vth(1,t) = abs(str2double(Rac3{1}(index + 101)));
            Vds(1,t) = abs(str2double(Rac3{1}(index + 79)));
%             if ((Vgs(1,t)) < (Vth(1,t))-0.05 || (Vds(1,t)) < (Vgs(1,t)-(Vth(1,t))))
%                 cut_off(1,i)=cut_off(1,i)+1;
%             end
%              if ((Vds(1,t)) < (Vgs(1,t)-(Vth(1,t))))
%                 triode(1,i)=triode(1,i)+1;
%             end
        end
 
    fid3 = fopen('amp.ma0');
    Rac3 = textscan(fid3, '%s','headerLines', 3);
    fclose(fid3);
  
    for j=1:4;
        out(1,j)= str2double(char(Rac3{1}(j)));
    end
    fid2 = fopen('amp.mt0');
    Rac2 = textscan(fid2, '%s','headerLines', 3);
    fclose(fid2);
 
    for j=1:2;
        out(1,j+4)= str2double(char(Rac2{1}(j)));
    end
    
    out(1,2)=abs(out(1,2));
    bw(1,i)=out(1,1);
    gain(1,i)=(out(1,2));
    if(out(1,3)>0 && out(1,4)>0)
           pm(1,i)=(atan(out(1,3)/out(1,4))*180/pi);     
        elseif(out(1,3)>0 && out(1,4)<0)
           pm(1,i)=0.1;%180-(atan(out(1,3)/out(1,4))*180/pi);
         elseif(out(1,3)<0 && out(1,4)<0)
            pm(1,i)=atan(out(1,3)/out(1,4))*180/pi;%180+(-1*(atan(out(1,3)/out(1,4))/pi*180));
        else
            pm(1,i)=10;
        end
        out(1,4)=out(1,5);
        out(1,5)=out(1,6);
        power(1,i)=out(1,4);
        area(1,i)=out(1,5);
        out(1,3)=pm(1,i); 

  if (cut_off(1,i) ~= 0 || triode(1,i) ~= 0)
        pm(1,i) = 10;
  end
  Vth_(i,:) = Vth;
  Vgs_(i,:) = Vgs;
  Vds_(i,:) = Vds;
end
end

