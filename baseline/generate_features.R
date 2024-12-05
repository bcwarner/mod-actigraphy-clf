###########################
#####  MOD1000 Daily  #####
###########################
rm(list=ls())
library(lubridate)
library(doParallel)
library(stringr)

filenames <- list.files("MOD Theme 3 Project/MOD 1000 Woman Activity Data",
                        pattern=".csv",
                        full.names=T,
                        recursive = T)

length(filenames)
#2726

names2 <- list.files("MOD Theme 3 Project/MOD 1000 Woman Activity Data",
                     pattern=".csv",
                     full.names=F,
                     recursive = T)
# names2 <- names2[newlist]
# filenames <- filenames[newlist]
names <- character(length(names2))
for(i in 1:length(names2)){
  i1 <- max(gregexpr('/',names2[i])[[1]])+1
  i2 <- nchar(names2[i])-4
  names[i] <- str_sub(names2[i],i1,i2)
}

length(unique(substr(names,1,4)))
#1278

All_IDs <- unique(substr(names,1,4))
Live_list <- read.csv('Chronotype/Live_birth.csv')[,2]
length(which(All_IDs %in% Live_list))

zeros <- function(x){length(which(x==0))}
zeros2 <- function(x){
  x <- c(1,x,1)
  empty <- c()
  for(i in 1:1435){
    if(x[i]!=0 & x[i+1]==0){
      start <- i+1
      end <- min(which(x[(start+1):1442]>0)+i)
      empty <- c(empty,end-start+1)
    }
  }
  return(max(empty))
}

library(foreach)
library(doParallel)
cores=detectCores()
cl <- makeCluster(cores[1]-1) #not to overload your computer
registerDoParallel(cl)

#Calculate the median mode of all days

for(n in 1:length(names)){
  PID <- names[n]
  print(PID)
  PatData <- read.csv(filenames[n],header = F,skip=20)
  if(ncol(PatData)>5 | ncol(PatData)<3)next
  PatData$V5 <- paste(PatData$V1,PatData$V2)
  PatData$V1 <- PatData$V2 <- NULL
  PatData$V5 <- strptime(PatData$V5, "%m/%d/%Y %I:%M:%S %p",tz="UTC")
  PatData$V3 <- as.numeric(as.character(PatData$V3))
  PatData$time <- PatData$V5
  PatData$MV <- PatData$V3
  PatData$MV[is.na(PatData$MV)==T] <- 0
  # if(is.null(PatData$V4)==F)PatData$V4 <- as.numeric(as.character(PatData$V4))
  # if(is.null(PatData$V4)==T)PatData$V4 <- numeric(nrow(PatData))
  noon <- which(hour(PatData$V5)==12 & minute(PatData$V5)==0)
  if(length(noon)<=1){print(paste(PID,'empty!'));next}
  if(tail(noon,1)+1439>nrow(PatData))noon <- noon[-length(noon)]
  timelist <- as.character(PatData$V5[noon+720])
  nightlist <- which(hour(PatData$time)==0 & minute(PatData$time)==0)
  
  WDs <- wday(PatData$time[noon],label = TRUE)
  
  WDlist <- which(WDs %in% c('Wed','Thu','Sun','Mon','Tue'))
  
  days <- length(noon)-1
  imagetable <- matrix(PatData$MV[noon[1]:(tail(noon,n = 1)-1)],ncol = length(noon)-1,nrow = 1440)
  imagetable[which(is.na(imagetable)==T,arr.ind = T)] <- 0
  goodlist <- which(apply(imagetable,2,zeros)<840)
  goodlist2 <- which(apply(imagetable,2,zeros2)<180)
  index <- intersect(goodlist,goodlist2)
  index <- intersect(index,WDlist)
  if(length(index)<=1)next
  imagetable <- imagetable[,index]
  
  # wday(PatData$time[i],label = TRUE)
  
  if(length(index)>=10){
    mypath <- file.path("Actigraphy/Heatmap_Weekday",paste(names[n],".jpg", sep = ""))
    jpeg(file = mypath, width = 800, height = 1200)
    par(mfrow=c(3,1),mar=c(3,4,2,2))
    image(imagetable[,ncol(imagetable):1], col  = gray((32:0)/32),xaxt = "n",yaxt = "n",main='Activity Data')
    axis(1,at=seq(0,1,by=0.125), lab=c("12PM","3PM","6PM","9PM","12AM","3AM","6AM","9AM","12PM"),cex.axis=0.9)
    # axis(2,at=seq(1,0,by=-1/(days-1)),lab=date(PatData$time)[noon[-1]],cex.axis=0.7,las=1)
    axis(2,at=seq(0,1,by=1/(length(index)-1)),lab=date(PatData$time[noon[index]]),cex.axis=0.7,las=1)
    
    mediantable <- matrix(rep(0,ncol(imagetable)*96),ncol=ncol(imagetable))
    for(i in 1:nrow(mediantable)){
      mediantable[i,] <- colMeans(imagetable[(i*15-14):(i*15),])
    }
    
    median_data <- apply(mediantable,1,median,na.rm=F)
    plot(median_data,type = 'l',ylab = "Activity",main = 'Dominant Activity Mode (Median)',
         xaxt = "n",cex.lab=1.5,cex.main=1.5,ylim = c(0,1000))
    sd_data <- apply(mediantable,1,IQR,na.rm=F)
    plot(sd_data,type = 'l',ylab = "Activity",main = 'Dominant Activity Mode (IQR)',
         xaxt = "n",cex.lab=1.5,cex.main=1.5,ylim = c(0,500))
    
    mode_data <- data.frame(rbind(median_data,sd_data))
    dev.off()
    setwd("Actigraphy/Heatmap_Weekday")
    write.csv(mode_data,paste(names[n],".csv", sep = ""))
  }
  
}





#459 460 467 473 485 521/522
for(n in 1:1000){
  PID <- names[n]
  print(PID)
  PatData <- read.csv(filenames[n],header = F,skip=20)
  if(ncol(PatData)>5 | ncol(PatData)<3)next
  PatData$V5 <- paste(PatData$V1,PatData$V2)
  PatData$V1 <- PatData$V2 <- NULL
  PatData$V5 <- strptime(PatData$V5, "%m/%d/%Y %I:%M:%S %p",tz="UTC")
  PatData$V3 <- as.numeric(as.character(PatData$V3))
  PatData$time <- PatData$V5
  PatData$MV <- PatData$V3
  PatData$MV[is.na(PatData$MV)==T] <- 0
  # if(is.null(PatData$V4)==F)PatData$V4 <- as.numeric(as.character(PatData$V4))
  # if(is.null(PatData$V4)==T)PatData$V4 <- numeric(nrow(PatData))
  noon <- which(hour(PatData$V5)==12 & minute(PatData$V5)==0)
  if(length(noon)<=1){print(paste(PID,'empty!'));next}
  if(tail(noon,1)+1439>nrow(PatData))noon <- noon[-length(noon)]
  timelist <- as.character(PatData$V5[noon+720])
  nightlist <- which(hour(PatData$time)==0 & minute(PatData$time)==0)
  
  days <- length(noon)-1
  imagetable <- matrix(log(PatData$MV[noon[1]:(tail(noon,n = 1)-1)]+1),ncol = length(noon)-1,nrow = 1440)
  imagetable[which(is.na(imagetable)==T,arr.ind = T)] <- 0
  goodlist <- which(apply(imagetable,2,zeros)<720)
  goodlist2 <- which(apply(imagetable,2,zeros2)<120)
  index <- intersect(goodlist,goodlist2)
  if(length(index)<=1)next
  imagetable <- imagetable[,index]
  
  mypath <- file.path("Actigraphy/Heatmap",paste(names[n],".jpg", sep = ""))
  jpeg(file = mypath, width = 800, height = 1200)
  par(mfrow=c(2,1),mar=c(3,4,2,2))
  image(imagetable[,ncol(imagetable):1], col  = gray((32:0)/32),xaxt = "n",yaxt = "n",main='Activity Data')
  axis(1,at=seq(0,1,by=0.125), lab=c("12PM","3PM","6PM","9PM","12AM","3AM","6AM","9AM","12PM"),cex.axis=0.9)
  # axis(2,at=seq(1,0,by=-1/(days-1)),lab=date(PatData$time)[noon[-1]],cex.axis=0.7,las=1)
  axis(2,at=seq(0,1,by=1/(length(index)-1)),lab=date(PatData$time[noon[index]]),cex.axis=0.7,las=1)
  
  mediandata <- apply(imagetable,1,median,na.rm=F)
  plot(exp(mediandata),type = 'l',ylab = "Activity",main = 'Dominant Activity Mode (Median)',
       xaxt = "n",cex.lab=1.5,cex.main=1.5,ylim = c(0,1000))
  
  dev.off()
  
  sleep_start <- c()
  sleep_end <- c()
  sleep_mid <- c()
  sleep_duration <- c()
  total_activity <- c()
  sleep_activity <- c()
  sleep_frequency <- c()
  weekday <- c()
  nightname <- c()
  zerolength <- c()
  
  for(i in nightlist){
    if(i <= 720 | i > nrow(PatData)-720)next
    temp <- PatData[(i-720):(i+719),]
    nightid <- date(temp$time[720])
    TT <- PatData$MV[(i-720):(i+719)]
    Height <- median(temp$MV[temp$MV>0])
    WD <- wday(PatData$time[i],label = TRUE)
    
    noise <- quantile(TT[TT>0],0.01)
    TT[TT<noise] <- 0
    TT[TT>=noise] <- 1
    
    # TT[TT>0] <- 1
    # for(j in 2:1439){TT[j] <- median(TT[(j-1):(j+1)])}
    if(length(which(TT==0)) >1000 | length(which(TT==1)) > 1200)next
    
    t10 <- numeric(1440/10)
    for(j in 1:144){
      t10[j] <- mean(TT[(1+10*j-10):(1+10*j-1)])
    }
    if(var(t10)==0)next
    alllocation <- expand.grid(seq(30,142),seq(10,142))
    L <- alllocation[alllocation$Var1+alllocation$Var2 < 144,]
    L$Var3 <- 144-L$Var1-L$Var2
    
    t10[ which(t10[1:54]==0)] <- 1
    t10[ which(t10[132:144]==0)] <- 1
    result <- foreach(k = 1:nrow(L), .combine = c) %dopar% cor(t10,c(rep(1,L[k,1]),rep(0,L[k,3]),rep(1,L[k,2])))
    l1 <- L[which.max(result),1]
    l2 <- 144-L[which.max(result),2]
    
    if(l2-l1<20)next
    alllocation <- expand.grid(seq(1:72),seq(1:72))
    L <- alllocation
    L$Var1 <- l1*10-36+L$Var1
    L$Var2 <- (144-l2)*10-36+L$Var2
    L$Var3 <- 1440-L$Var1-L$Var2
    result <- foreach(k = 1:nrow(L), .combine = c) %dopar% cor(TT,c(rep(1,L[k,1]),rep(0,L[k,3]),rep(1,L[k,2])))
    l1 <- L[which.max(result),1]
    l2 <- L[which.max(result),1]+L[which.max(result),3]+1
    if(sum(temp$MV[1:l1])==0)next
    if(sum(temp$MV[l2:1440])==0)next
    if(temp$MV[l1]==0){for(j in 1:300){if(temp$MV[l1-j]>0){l1 <- l1-j;break}}}
    if(temp$MV[l2]==0){for(j in 1:min(100,1440-l2)){if(temp$MV[l2+j]>0){l2 <- l2+j;break}}}
    
    print(paste(PID,nightid))
    sleep_start <- c(sleep_start,l1)
    sleep_end <- c(sleep_end,l2)
    sleep_mid <- c(sleep_mid,as.integer((l1+l2)/2))
    sleep_duration <- c(sleep_duration,(l2-l1))
    total_activity <- c(total_activity,sum(temp$MV))
    sleep_activity <- c(sleep_activity,sum(temp$MV[l1:(l2-1)]))
    sleep_frequency <- c(sleep_frequency,length(which(temp$MV[l1:(l2-1)]>0))/(l2-l1))
    weekday <- c(weekday,WD)
    nightname <- c(nightname,as.character(nightid))
    zerolength <- c(zerolength,zeros2(PatData$MV[(i-720):(i+719)]))
    
    mypath <- file.path("Actigraphy/DailyResults",paste(PID,nightid,".jpg", sep = ""))
    jpeg(file = mypath, width = 800, height = 480)
    plot(temp$MV,type="l",xlab="",ylab="",xaxt="n")
    axis(1,at=seq(1,1441,by=180), lab=c("12PM","3PM","6PM","9PM","12AM","3AM","6AM","9AM","12PM"),cex=0.8 )
    lines(6*c(rep(Height,l1),rep(0,(l2-l1)),rep(Height,(1440-l2))),col="blue",type="l",lwd=2)
    dev.off()
    
    
  }
  if(length(sleep_start)>0){
    PATname <- rep(PID,length(sleep_start))
    sleep_result <- as.data.frame(cbind(PATname,sleep_start,sleep_end,sleep_mid,sleep_duration,weekday,total_activity,sleep_activity,sleep_frequency,nightname,zerolength))
    write.csv(sleep_result,paste("Actigraphy/DailyResults/Result_",PID,".csv",sep = ""),row.names = F)
    
  }
  
  #Total_result <- rbind(Total_result,sleep_result)
  #closeAllConnections()
}

csv_list <- list.files("MOD1000DailyPlots/NewResults",
                         pattern="*\\.csv",
                         full.names=T,
                         recursive = T)
file.copy(csv_list,"MOD1000DailyPlots/CSVs")







Total_result <- data.frame()
resultlist <- list.files("MOD1000DailyPlots/CSVs",
                         pattern="*\\.csv",
                         full.names=T,
                         recursive = T)
for(n in 1:length(resultlist)){
  temp <- read.csv(resultlist[n])
  Total_result <- rbind(Total_result,temp)
}

Total_result$ID <- substr(Total_result$PATname,1,4)
Total_result$ID[which(Total_result$ID=='P127')] <- '1276'
dim(Total_result)
# 37888    12
write.csv(Total_result,'Chronotype/MOD1000Daily_20200717.csv',row.names = F)

Total_result <- read.csv('Chronotype/MOD1000Daily_20200717.csv')

##################################




##################################


# find the best cutoff for zerolengths
cutoffs <- seq(5,200,5)
pass_num <- numeric(length(cutoffs))
for(i in 1:length(cutoffs))pass_num[i] <- length(which(Total_result$zerolength<cutoffs[i]))
par(mfrow=c(2,1))
plot(cutoffs,pass_num,t='l',main = 'Consecutive missing minutes cutoff',ylab = 'number of days');abline(v=90)
delta_pass <- (c(pass_num,0)-c(0,pass_num))[-41]
plot(cutoffs,delta_pass,t='l')


MOD <- read.csv('Grant/Circadian/MOD1000WomenCohort_DATA_2020-06-29_1412.csv')
EDC <- MOD[,c("record_id",'ptb_37wks','sptb_37wks','ga_wks','ga_days','edc')]
colnames(EDC) <- c('ID','PTB','SPTB','GAWeek','GADay','Edc')

# Filter the passdata
#remove two duplicated files
# Total_result <- Total_result[-which(Total_result$PATname%in% c('2471_GA9','1732_GA16')),]

passdata <- Total_result[which(Total_result$zerolength<90),]
passdata <- merge(passdata,EDC)
passdata$nightname <- ymd(passdata$nightname)
passdata$Edc <- ymd(passdata$Edc)
passdata$Deliver <- passdata$Edc-280+7*passdata$GAWeek+passdata$GADay

#Correction
# passdata$Edc[which(passdata$ID=='1018')] <- passdata$Deliver[which(passdata$ID=='1018')]-37*7+280
# passdata$Edc[which(passdata$ID=='2073')] <- passdata$Deliver[which(passdata$ID=='2073')]-39*7+280
# passdata <- passdata[-which(passdata$ID=='1288'),]
# passdata <- passdata[-which(passdata$ID=='1276'),]
# passdata <- passdata[-which(passdata$Deliver <= ymd('20000101')),]

hist(passdata$Deliver,breaks = 30)
range((passdata$nightname-(passdata$Edc-280))/7,na.rm = T)
passdata[which(passdata$nightname-(passdata$Edc-280)<0),-(4:10)]

passdata$Trim <- rep(3,nrow(passdata))
passdata$Trim[which(passdata$nightname-(passdata$Edc-280)<=181)] <- 2
# passdata$Trim[which(passdata$nightname-(passdata$Edc-280)<=188)] <- 2
passdata$Trim[which(passdata$nightname-(passdata$Edc-280)<=90)] <- 1

passdata$GA_Days <- as.numeric(passdata$nightname-(passdata$Edc-280))

passdata <- passdata[which(passdata$GA_Days>0),]
passdata <- passdata[order(passdata$ID, passdata$nightname),]

passdata$ID_Trim <- paste(passdata$ID,passdata$Trim,sep = '_')
passdata$ddonset <- numeric(nrow(passdata))
passdata$ddoffset <- numeric(nrow(passdata))
passdata$ddmid <- numeric(nrow(passdata))
passdata$ddduration <- numeric(nrow(passdata))
passdata$ndays <- numeric(nrow(passdata))

for(i in unique(passdata$ID_Trim)){
  temp <- passdata[passdata$ID_Trim==i,]
  temp$ndays <- nrow(temp)
  if(nrow(temp)>1){
    for(j in 2:nrow(temp)){
      temp$ddonset[j] <- temp$sleep_start[j]-temp$sleep_start[j-1]
      temp$ddoffset[j] <- temp$sleep_end[j]-temp$sleep_end[j-1]
      temp$ddmid[j] <- temp$sleep_mid[j]-temp$sleep_mid[j-1]
      temp$ddduration[j] <- temp$sleep_duration[j]-temp$sleep_duration[j-1]
    }
  }
  passdata[passdata$ID_Trim==i,] <- temp
}

dim(passdata)


# find the best cutoff for number of days
cutoffs <- seq(1,30)
pass_num <- numeric(length(cutoffs))
for(i in 1:length(cutoffs))pass_num[i] <- nrow(passdata[which(passdata$ndays>cutoffs[i]),])
plot(cutoffs,pass_num,t='l',ylim = c(1,30000),ylab = 'number of days',main = 'Number of days cutoff')
abline(v=10)

passdata <- passdata[which(passdata$ndays>10),]
workpassdata <- passdata[passdata$weekday %in% c(2:6),]

unique(workpassdata$ID)
length(unique(workpassdata$ID))
#678 patients
write.csv(passdata,'Chronotype/MOD1000passdata_20200729.csv')

passdata <- read.csv('Chronotype/MOD1000passdata_20200729.csv')
passdata[1:3,]
dim(passdata)
length(unique(passdata$ID))



MedianTable <- aggregate(sleep_start~ID+Trim, passdata[which(passdata$weekday %in% 2:6),], median)
MedianTable <- cbind(MedianTable,
                     sleepend=aggregate(sleep_end~ID+Trim, passdata[which(passdata$weekday %in% c(2:6)),], median)[,3],
                     sleepmid=aggregate(sleep_mid~ID+Trim, passdata[which(passdata$weekday %in% c(2:6)),], median)[,3],
                     sleepduration=aggregate(sleep_duration~ID+Trim, passdata[which(passdata$weekday %in% c(2:6)),], median)[,3],
                     totalactivity=aggregate(total_activity~ID+Trim, passdata[which(passdata$weekday %in% c(2:6)),], median)[,3],
                     sleepactivity=aggregate(sleep_activity~ID+Trim, passdata[which(passdata$weekday %in% c(2:6)),], median)[,3],
                     sleepfrequency=aggregate(sleep_frequency~ID+Trim, passdata[which(passdata$weekday %in% c(2:6)),], median)[,3],
                     ddsleepstart=aggregate(abs(ddonset)~ID+Trim, passdata[which(passdata$weekday %in% c(2:6)),], median)[,3],
                     ddsleepend=aggregate(abs(ddoffset)~ID+Trim, passdata[which(passdata$weekday %in% c(2:6)),], median)[,3],
                     ddsleepmid=aggregate(abs(ddmid)~ID+Trim, passdata[which(passdata$weekday %in% c(2:6)),], median)[,3],
                     ddsleepduration=aggregate(abs(ddduration)~ID+Trim, passdata[which(passdata$weekday %in% c(2:6)),], median)[,3]
)


MedianTable <- MedianTable[order(MedianTable$ID,MedianTable$Trim),]

MedianTable <- merge(MedianTable,EDC,by = 'ID')
MedianTable$GA_Days <- MedianTable$GAWeek*7+MedianTable$GADay

write.csv(MedianTable,'Chronotype/MOD1000Median_20200729.csv',row.names = F)

MedianTable <- read.csv('Chronotype/MOD1000Median_20200729.csv')
dim(MedianTable)
str(MedianTable)
names(MedianTable)
MedianTable$Edc <- ymd(MedianTable$Edc)
# MedianTable$Deliver <- ymd(MedianTable$Deliver)

length(unique(MedianTable$ID))
#678 patients

ID_list <- unique(MedianTable$ID)
names(MedianTable)[1] <- 'record_id'
MOD <- read.csv('Chronotype/MOD1000WomenCohort_DATA_2021-09-27_0959.csv')
MOD <- MOD[which(MOD$redcap_event_name=='general_arm_1'),]
dim(MOD)
table(MOD$intrauter_fetal_demise)

MedianTable <- merge(MedianTable,MOD[,c('record_id','intrauter_fetal_demise')],all.x = T)

table(MedianTable$intrauter_fetal_demise)

##############
names(MedianTable)
boxplot(sleep_start~intrauter_fetal_demise,data=MedianTable)
boxplot(sleepend~intrauter_fetal_demise,data=MedianTable)
boxplot(sleepduration~intrauter_fetal_demise,data=MedianTable)
boxplot(totalactivity~intrauter_fetal_demise,data=MedianTable)
boxplot(sleepactivity~intrauter_fetal_demise,data=MedianTable)
boxplot(sleepfrequency~intrauter_fetal_demise,data=MedianTable)

boxplot(ddsleepstart~intrauter_fetal_demise,data=MedianTable)
boxplot(ddsleepend~intrauter_fetal_demise,data=MedianTable)
boxplot(ddsleepduration~intrauter_fetal_demise,data=MedianTable)


##############
#Does sleep variable change across pregnancy?
boxplot(ddsleepstart~Trim,data=MedianTable)
boxplot(ddsleepend~Trim,data=MedianTable)
boxplot(sleep_start~Trim,data=MedianTable)
boxplot(sleepend~Trim,data=MedianTable)


###############Demographic
list_123 <- c()
for(i in unique(MedianTable$ID)){
  if(1 %in% MedianTable$Trim[MedianTable$ID==i] &
     2 %in% MedianTable$Trim[MedianTable$ID==i] &
     3 %in% MedianTable$Trim[MedianTable$ID==i])list_123 <- c(list_123,i)
}
length(list_123)

list_12 <- c()
for(i in unique(MedianTable$ID)){
  if(1 %in% MedianTable$Trim[MedianTable$ID==i] &
     2 %in% MedianTable$Trim[MedianTable$ID==i])list_12 <- c(list_12,i)
}
list_12 <- setdiff(list_12,list_123)
length(list_12)

list_13 <- c()
for(i in unique(MedianTable$ID)){
  if(1 %in% MedianTable$Trim[MedianTable$ID==i] &
     3 %in% MedianTable$Trim[MedianTable$ID==i])list_13 <- c(list_13,i)
}
list_13 <- setdiff(list_13,list_123)
length(list_13)

list_23 <- c()
for(i in unique(MedianTable$ID)){
  if(2 %in% MedianTable$Trim[MedianTable$ID==i] &
     3 %in% MedianTable$Trim[MedianTable$ID==i])list_23 <- c(list_23,i)
}
list_23 <- setdiff(list_23,list_123)
length(list_23)

list_1 <- c()
for(i in unique(MedianTable$ID)){
  if(1 %in% MedianTable$Trim[MedianTable$ID==i]
     & length(unique(MedianTable$Trim[MedianTable$ID==i]))==1)list_1 <- c(list_1,i)
}
length(list_1)
#97

list_2 <- c()
for(i in unique(MedianTable$ID)){
  if(2 %in% MedianTable$Trim[MedianTable$ID==i]
     & length(unique(MedianTable$Trim[MedianTable$ID==i]))==1)list_2 <- c(list_2,i)
}
length(list_2)
#122


list_3 <- c()
for(i in unique(MedianTable$ID)){
  if(3 %in% MedianTable$Trim[MedianTable$ID==i]
     & length(unique(MedianTable$Trim[MedianTable$ID==i]))==1)list_3 <- c(list_3,i)
}
length(list_3)

list_1_all <- c()
for(i in unique(MedianTable$ID)){
  if(1 %in% MedianTable$Trim[MedianTable$ID==i]
     )list_1_all <- c(list_1_all,i)
}
length(list_1_all)

list_2_all <- c()
for(i in unique(MedianTable$ID)){
  if(2 %in% MedianTable$Trim[MedianTable$ID==i]
  )list_2_all <- c(list_2_all,i)
}
length(list_2_all)

list_3_all <- c()
for(i in unique(MedianTable$ID)){
  if(3 %in% MedianTable$Trim[MedianTable$ID==i]
  )list_3_all <- c(list_3_all,i)
}
length(list_3_all)

list_12_all <- c()
for(i in unique(MedianTable$ID)){
  if(1 %in% MedianTable$Trim[MedianTable$ID==i] | 2 %in% MedianTable$Trim[MedianTable$ID==i])list_12_all <- c(list_12_all,i)
}
length(list_12_all)
#625

#############
### Extract values from T1/T2
passdata12 <- passdata[which(passdata$Trim<=2),]

MedianTable12 <- aggregate(sleep_start~ID, passdata12[which(passdata12$weekday %in% 2:6),], median)
MedianTable12 <- cbind(MedianTable12,
                     sleepend=aggregate(sleep_end~ID, passdata12[which(passdata12$weekday %in% c(2:6)),], median)[,2],
                     sleepmid=aggregate(sleep_mid~ID, passdata12[which(passdata12$weekday %in% c(2:6)),], median)[,2],
                     sleepduration=aggregate(sleep_duration~ID, passdata12[which(passdata12$weekday %in% c(2:6)),], median)[,2],
                     totalactivity=aggregate(total_activity~ID, passdata12[which(passdata12$weekday %in% c(2:6)),], median)[,2],
                     sleepactivity=aggregate(sleep_activity~ID, passdata12[which(passdata12$weekday %in% c(2:6)),], median)[,2],
                     sleepfrequency=aggregate(sleep_frequency~ID, passdata12[which(passdata12$weekday %in% c(2:6)),], median)[,2],
                     ddsleepstart=aggregate(abs(ddonset)~ID, passdata12[which(passdata12$weekday %in% c(2:6)),], median)[,2],
                     ddsleepend=aggregate(abs(ddoffset)~ID, passdata12[which(passdata12$weekday %in% c(2:6)),], median)[,2],
                     ddsleepmid=aggregate(abs(ddmid)~ID, passdata12[which(passdata12$weekday %in% c(2:6)),], median)[,2],
                     ddsleepduration=aggregate(abs(ddduration)~ID, passdata12[which(passdata12$weekday %in% c(2:6)),], median)[,2]
)


MedianTable12 <- MedianTable12[order(MedianTable12$ID),]
dim(MedianTable12)
colnames(MedianTable12)[1] <- 'record_id'
# MedianTable <- merge(MedianTable,EDC,by = 'ID')
# MedianTable$GA_Days <- MedianTable$GAWeek*7+MedianTable$GADay








MOD2 <- MOD[which(MOD$record_id %in% list_12_all),]
MOD2$GAW <- MOD2$ga_wks+MOD2$ga_days/7
hist(MOD2$GAW)
dim(MOD2)
MOD2 <- MOD2[which(MOD2$GAW>0),]
dim(MOD2)

MOD2 <- merge(MOD2,MedianTable12,by='record_id',all.x = T)
dim(MOD2)

hist(MOD2$age_enroll)
summary(MOD2$age_enroll)

plot(MOD2$age_enroll,MOD2$ddsleepduration)

t.test(MOD2$ddsleepstart[MOD2$age_enroll<=29.5]/60,MOD2$ddsleepstart[MOD2$age_enroll>29.5]/60)
mean(MOD2$ddsleepstart[MOD2$age_enroll<=29.5]/60)

t.test(MOD2$ddsleepstart[MOD2$race==1]/60,MOD2$ddsleepstart[MOD2$race==2]/60)
mean(MOD2$ddsleepstart[MOD2$race==1])/60-mean(MOD2$ddsleepstart[MOD2$race==2])/60

t.test(MOD2$ddsleepstart[MOD2$race>2]/60,MOD2$ddsleepstart[MOD2$race==2]/60)
mean(MOD2$ddsleepstart[MOD2$race>2])/60-mean(MOD2$ddsleepstart[MOD2$race==2])/60

MOD2$new_race <- 0;MOD2$new_race[which(MOD2$race==1)] <- 1;MOD2$new_race[which(MOD2$race>2 | is.na(MOD2$race)==T)] <- 2
MOD2$new_race <- as.factor(MOD2$new_race)
table(MOD2$new_race)

t.test(MOD2$ddsleepstart[MOD2$new_race==0]/60,MOD2$ddsleepstart[MOD2$new_race==2]/60)
mean(MOD2$ddsleepstart[MOD2$new_race==0]/60)-mean(MOD2$ddsleepstart[MOD2$new_race==2]/60)

library(tidyverse)
library(rstatix)
library(ggpubr)

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ new_race)
res.aov

MOD2$education[which(MOD2$education==-99)] <- NA

#EDU
MOD2$new_edu <- NA
MOD2$new_edu[which(MOD2$education %in% 1:2)] <- 0
MOD2$new_edu[which(MOD2$education == 3)] <- 1
MOD2$new_edu[which(MOD2$education == 4)] <- 2
MOD2$new_edu <- as.factor(MOD2$new_edu)

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ new_edu)
res.aov

t.test(MOD2$ddsleepstart[MOD2$new_edu==1]/60,MOD2$ddsleepstart[MOD2$new_edu==0]/60)
mean(MOD2$ddsleepstart[MOD2$new_edu==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$new_edu==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$new_edu==2]/60,MOD2$ddsleepstart[MOD2$new_edu==0]/60)
mean(MOD2$ddsleepstart[MOD2$new_edu==2]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$new_edu==0]/60,na.rm = T)

#MARITAL
MOD2$new_marital <- 2
MOD2$new_marital[which(MOD2$marital == 1)] <- 0
MOD2$new_marital[which(MOD2$marital == 2)] <- 1
MOD2$new_marital <- as.factor(MOD2$new_marital)

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ new_marital)
res.aov

t.test(MOD2$ddsleepstart[MOD2$new_marital==1]/60,MOD2$ddsleepstart[MOD2$new_marital==0]/60)
mean(MOD2$ddsleepstart[MOD2$new_marital==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$new_marital==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$new_marital==2]/60,MOD2$ddsleepstart[MOD2$new_marital==0]/60)
mean(MOD2$ddsleepstart[MOD2$new_marital==2]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$new_marital==0]/60,na.rm = T)

#EMP
MOD2$new_employ <- 0
MOD2$new_employ[which(MOD2$employed == 1)] <- 1
MOD2$new_employ[which(MOD2$employed == 3)] <- 2
MOD2$new_employ[which(MOD2$employed == 4)] <- 3
MOD2$new_employ <- as.factor(MOD2$new_employ)
table(MOD2$new_employ)

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ new_employ)
res.aov

t.test(MOD2$ddsleepstart[MOD2$new_employ==1]/60,MOD2$ddsleepstart[MOD2$new_employ==0]/60)
mean(MOD2$ddsleepstart[MOD2$new_employ==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$new_employ==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$new_employ==2]/60,MOD2$ddsleepstart[MOD2$new_employ==0]/60)
mean(MOD2$ddsleepstart[MOD2$new_employ==2]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$new_employ==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$new_employ==3]/60,MOD2$ddsleepstart[MOD2$new_employ==0]/60)
mean(MOD2$ddsleepstart[MOD2$new_employ==3]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$new_employ==0]/60,na.rm = T)

#shift
table(MOD2$nightshift_1trim)
table(MOD2$nightshift_2trim)
table(MOD2$nightshift_3trim)
MOD2$Shift <- 0
MOD2$Shift[which(MOD2$nightshift_1trim==1|MOD2$nightshift_2trim==1|MOD2$nightshift_3trim==1)] <- 1
table(MOD2$Shift)

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ Shift)
res.aov

t.test(MOD2$ddsleepstart[MOD2$Shift==1]/60,MOD2$ddsleepstart[MOD2$Shift==0]/60)
mean(MOD2$ddsleepstart[MOD2$Shift==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$Shift==0]/60,na.rm = T)

#Insurance
table(MOD2$insur)

MOD2$Insurance <- 0
MOD2$Insurance[which(MOD2$insur %in% 1:2)] <- 1
MOD2$Insurance[which(MOD2$insur ==4)] <- 2
MOD2$Insurance[which(MOD2$insur %in% c(-99,5))] <- 3
table(MOD2$Insurance)
MOD2$Insurance <- as.factor(MOD2$Insurance)

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ Insurance)
res.aov

t.test(MOD2$ddsleepstart[MOD2$Insurance==1]/60,MOD2$ddsleepstart[MOD2$Insurance==0]/60)
mean(MOD2$ddsleepstart[MOD2$Insurance==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$Insurance==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$Insurance==2]/60,MOD2$ddsleepstart[MOD2$Insurance==0]/60)
mean(MOD2$ddsleepstart[MOD2$Insurance==2]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$Insurance==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$Insurance==3]/60,MOD2$ddsleepstart[MOD2$Insurance==0]/60)
mean(MOD2$ddsleepstart[MOD2$Insurance==3]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$Insurance==0]/60,na.rm = T)

#GA at delivery
length(which(MOD2$GAW>=37))
length(which(MOD2$GAW<37 & MOD2$GAW>=34))
length(which(MOD2$GAW<34))

MOD2$PTB <- 0
MOD2$PTB[which(MOD2$GAW<37 & MOD2$GAW>=34)] <- 1
MOD2$PTB[which(MOD2$GAW<34)] <- 2
MOD2$PTB <- as.factor(MOD2$PTB)
table(MOD2$PTB)

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ PTB)
res.aov

t.test(MOD2$ddsleepstart[MOD2$PTB==1]/60,MOD2$ddsleepstart[MOD2$PTB==0]/60)
mean(MOD2$ddsleepstart[MOD2$PTB==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$PTB==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$PTB==2]/60,MOD2$ddsleepstart[MOD2$PTB==0]/60)
mean(MOD2$ddsleepstart[MOD2$PTB==2]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$PTB==0]/60,na.rm = T)

#Tobacco
table(MOD2$smoke)
MOD2$smoke <- as.factor(MOD2$smoke)

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ smoke)
res.aov

t.test(MOD2$ddsleepstart[MOD2$smoke==1]/60,MOD2$ddsleepstart[MOD2$smoke==0]/60)
mean(MOD2$ddsleepstart[MOD2$smoke==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$smoke==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$smoke==2]/60,MOD2$ddsleepstart[MOD2$smoke==0]/60)
mean(MOD2$ddsleepstart[MOD2$smoke==2]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$smoke==0]/60,na.rm = T)

#Alcohol
table(MOD2$alcohol)
MOD2$alcohol <- as.factor(MOD2$alcohol)

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ alcohol)
res.aov

t.test(MOD2$ddsleepstart[MOD2$alcohol==1]/60,MOD2$ddsleepstart[MOD2$alcohol==0]/60)
mean(MOD2$ddsleepstart[MOD2$alcohol==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$alcohol==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$alcohol==2]/60,MOD2$ddsleepstart[MOD2$alcohol==0]/60)
mean(MOD2$ddsleepstart[MOD2$alcohol==2]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$alcohol==0]/60,na.rm = T)

#Drugs
table(MOD2$drugs)
MOD2$drugs <- as.factor(MOD2$drugs)

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ drugs)
res.aov

t.test(MOD2$ddsleepstart[MOD2$drugs==1]/60,MOD2$ddsleepstart[MOD2$drugs==0]/60)
mean(MOD2$ddsleepstart[MOD2$drugs==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$drugs==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$drugs==2]/60,MOD2$ddsleepstart[MOD2$drugs==0]/60)
mean(MOD2$ddsleepstart[MOD2$drugs==2]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$drugs==0]/60,na.rm = T)

#Income
table(MOD2$income_annual1)

MOD2$Income <- 0
MOD2$Income[which(MOD2$income_annual1 %in% 2:3)] <- 1
MOD2$Income[which(MOD2$income_annual1 %in% 4:5)] <- 2
MOD2$Income[which(MOD2$income_annual1 %in% 6:8)] <- 3
MOD2$Income[which(MOD2$income_annual1 %in% 9)] <- 4

MOD2$Income <- as.factor(MOD2$Income)
table(MOD2$Income)

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ Income)
res.aov

t.test(MOD2$ddsleepstart[MOD2$Income==0]/60,MOD2$ddsleepstart[MOD2$Income==4]/60)
mean(MOD2$ddsleepstart[MOD2$Income==0]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$Income==4]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$Income==1]/60,MOD2$ddsleepstart[MOD2$Income==4]/60)
mean(MOD2$ddsleepstart[MOD2$Income==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$Income==4]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$Income==2]/60,MOD2$ddsleepstart[MOD2$Income==4]/60)
mean(MOD2$ddsleepstart[MOD2$Income==2]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$Income==4]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$Income==3]/60,MOD2$ddsleepstart[MOD2$Income==4]/60)
mean(MOD2$ddsleepstart[MOD2$Income==3]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$Income==4]/60,na.rm = T)


#Nulli

table(MOD2$nullliparous)

MOD2$nulliparous <- MOD2$nullliparous

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ nulliparous)
res.aov

t.test(MOD2$ddsleepstart[MOD2$nulliparous==1]/60,MOD2$ddsleepstart[MOD2$nulliparous==0]/60)
mean(MOD2$ddsleepstart[MOD2$nulliparous==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$nulliparous==0]/60,na.rm = T)

#bmi
table(is.na(MOD2$bmi_1vis))

# <18.5 
# 18.5-24.9 ref
# 25-29.9
# 30-34.9
# ≥ 35
MOD2$BMI <- 0
MOD2$BMI[which(MOD2$bmi_1vis<18.5)] <- 1
MOD2$BMI[which(MOD2$bmi_1vis>=25 & MOD2$bmi_1vis<30)] <- 2
MOD2$BMI[which(MOD2$bmi_1vis>=30 & MOD2$bmi_1vis<35)] <- 3
MOD2$BMI[which(MOD2$bmi_1vis>=35)] <- 4

table(MOD2$BMI)


res.aov <- MOD2 %>% anova_test(ddsleepstart ~ BMI)
res.aov

t.test(MOD2$ddsleepstart[MOD2$BMI==1]/60,MOD2$ddsleepstart[MOD2$BMI==0]/60)
mean(MOD2$ddsleepstart[MOD2$BMI==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$BMI==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$BMI==2]/60,MOD2$ddsleepstart[MOD2$BMI==0]/60)
mean(MOD2$ddsleepstart[MOD2$BMI==2]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$BMI==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$BMI==3]/60,MOD2$ddsleepstart[MOD2$BMI==0]/60)
mean(MOD2$ddsleepstart[MOD2$BMI==3]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$BMI==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$BMI==4]/60,MOD2$ddsleepstart[MOD2$BMI==0]/60)
mean(MOD2$ddsleepstart[MOD2$BMI==4]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$BMI==0]/60,na.rm = T)


#Infant gender
table(MOD2$infant_gender)
table(is.na(MOD2$infant_gender))
MOD2$infant_gender <- as.factor(MOD2$infant_gender)
MOD2$infant_gender[which(is.na(MOD2$infant_gender)==T)] <- 4

res.aov <- MOD2[MOD2$infant_gender %in% 1:2,] %>% anova_test(ddsleepstart ~ infant_gender)
res.aov

t.test(MOD2$ddsleepstart[MOD2$infant_gender==2]/60,MOD2$ddsleepstart[MOD2$infant_gender==1]/60)
mean(MOD2$ddsleepstart[MOD2$infant_gender==2]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$infant_gender==1]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$infant_gender==4]/60,MOD2$ddsleepstart[MOD2$infant_gender==1]/60)
mean(MOD2$ddsleepstart[MOD2$infant_gender==4]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$infant_gender==1]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$infant_gender==4]/60,MOD2$ddsleepstart[MOD2$infant_gender==2]/60)
mean(MOD2$ddsleepstart[MOD2$infant_gender==4]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$infant_gender==2]/60,na.rm = T)

# ADI
table(MOD2$adi1)
MOD2$ADI <- as.numeric(as.character(MOD2$adi1))
hist(MOD2$ADI)

MOD2$ADI[which(MOD2$ADI<9)] <- 0
MOD2$ADI[which(MOD2$ADI>=9)] <- 1
MOD2$ADI[which(is.na(MOD2$ADI)==T)] <- 2
table(MOD2$ADI)
MOD2$ADI <- as.factor(MOD2$ADI)

res.aov <- MOD2 %>% anova_test(ddsleepstart ~ ADI)
res.aov

t.test(MOD2$ddsleepstart[MOD2$ADI==1]/60,MOD2$ddsleepstart[MOD2$ADI==0]/60)
mean(MOD2$ddsleepstart[MOD2$ADI==1]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$ADI==0]/60,na.rm = T)

t.test(MOD2$ddsleepstart[MOD2$ADI==2]/60,MOD2$ddsleepstart[MOD2$ADI==0]/60)
mean(MOD2$ddsleepstart[MOD2$ADI==2]/60,na.rm = T)-mean(MOD2$ddsleepstart[MOD2$ADI==0]/60,na.rm = T)

##############
#Outcome
MOD2$PTB34 <- 0; MOD2$PTB34[which(MOD2$GAW<34)] <- 1
table(MOD2$PTB34)

MOD2$PTB37 <- 0; MOD2$PTB37[which(MOD2$GAW<37)] <- 1
table(MOD2$PTB37)

hist(MOD2$ddsleepstart)
summary(MOD2$ddsleepstart)

# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.00   40.00   59.00   65.79   83.25  215.00 

MOD2$ddsleepstart_quarter <- 3
MOD2$ddsleepstart_quarter[which(MOD2$ddsleepstart<=83.25)] <- 2
MOD2$ddsleepstart_quarter[which(MOD2$ddsleepstart<=59)] <- 1
MOD2$ddsleepstart_quarter[which(MOD2$ddsleepstart<=40)] <- 0
table(MOD2$ddsleepstart_quarter)

MOD2$ddsleepstart_low <- 0
MOD2$ddsleepstart_low[which(MOD2$ddsleepstart<=59)] <- 1
table(MOD2$ddsleepstart_low)

MOD2$ddsleepstart_Q1 <- 1
MOD2$ddsleepstart_Q1[which(MOD2$ddsleepstart<=40)] <- 0
table(MOD2$ddsleepstart_Q1)

MOD2$ddsleepstart_Q4 <- 1
MOD2$ddsleepstart_Q4[which(MOD2$ddsleepstart<=83.25)] <- 0
table(MOD2$ddsleepstart_Q4)

table(MOD2$PTB34,MOD2$ddsleepstart_quarter)

table(MOD2$ddsleepstart_quarter,MOD2$PTB34)
table(MOD2$ddsleepstart_Q1,MOD2$PTB34)
table(MOD2$ddsleepstart_Q4,MOD2$PTB34)

table(MOD2$ddsleepstart_quarter,MOD2$PTB37)
table(MOD2$ddsleepstart_Q1,MOD2$PTB37)
table(MOD2$ddsleepstart_Q4,MOD2$PTB37)

library(epitools)
riskratio(table(MOD2$ddsleepstart_Q1,MOD2$PTB34), conf.level = 0.95,correction = T)
riskratio(table(MOD2$ddsleepstart_Q4,MOD2$PTB34), conf.level = 0.95,correction = T)

riskratio(table(MOD2$ddsleepstart_quarter,MOD2$PTB34), conf.level = 0.95,correction = T)

riskratio(table(MOD2$ddsleepstart_low,MOD2$PTB34), conf.level = 0.95,correction = T)
riskratio(table(MOD2$ddsleepstart_low,MOD2$PTB34), conf.level = 0.95,correction = F)


test <- epi.2by2(table(MOD2$ddsleepstart_low,MOD2$PTB34),method = 'cohort.count')


dat <- matrix(c(13,2163,5,3349), nrow = 2, byrow = TRUE)
rownames(dat) <- c("DF+", "DF-"); colnames(dat) <- c("FUS+", "FUS-") 
dat


MOD2$PTB34 <- 1; MOD2$PTB34[which(MOD2$GAW<34)] <- 0
table(MOD2$PTB34)
test <- epi.2by2(table(MOD2$ddsleepstart_low,MOD2$PTB34), method = "cohort.count", 
         conf.level = 0.95, units = 100, outcome = "as.columns")
test$massoc$RR.strata.wald

MOD2$PTB37 <- 1; MOD2$PTB37[which(MOD2$GAW<37)] <- 0
table(MOD2$PTB37)
test <- epi.2by2(table(MOD2$ddsleepstart_low,MOD2$PTB37), method = "cohort.count", 
                 conf.level = 0.95, units = 100, outcome = "as.columns")
test$massoc$RR.strata.wald

test <- epi.2by2(table(MOD2$ddsleepstart_quarter,MOD2$PTB34), method = "cohort.count", 
                 conf.level = 0.95, units = 100, outcome = "as.columns")
test$massoc$RR.strata.wald

MOD2$ddsleepstart_Q4 <- 0
MOD2$ddsleepstart_Q4[which(MOD2$ddsleepstart<=83.25)] <- 1
table(MOD2$ddsleepstart_Q4)
test <- epi.2by2(table(MOD2$ddsleepstart_Q4,MOD2$PTB34), method = "cohort.count", 
                 conf.level = 0.95, units = 100, outcome = "as.columns")
test$massoc$RR.strata.wald

MOD2$ddsleepstart_quarter <- as.factor(MOD2$ddsleepstart_quarter)


test <- epi.2by2(table(MOD2$ddsleepstart_Q4,MOD2$PTB37), method = "cohort.count", 
                 conf.level = 0.95, units = 100, outcome = "as.columns")
test$massoc$RR.strata.wald

table(is.na(MOD2$preeclampsia))
MOD2$preeclampsia[which(MOD2$preeclampsia==2)] <- NA

table(MOD3$preeclampsia)
MOD3$PreE <- 1;MOD3$PreE[which(MOD3$preeclampsia==1)] <- 0;
table(MOD3$PreE)
test <- epi.2by2(table(MOD3$ddsleepstart_low,MOD3$PreE), method = "cohort.count", 
                 conf.level = 0.95, units = 100, outcome = "as.columns")
test$massoc$RR.strata.wald

table(MOD3$PreE37)
MOD3$PreE_37 <- 1;MOD3$PreE_37[which(MOD3$PreE37==1)] <- 0;
table(MOD3$PreE_37)
test <- epi.2by2(table(MOD3$ddsleepstart_low,MOD3$PreE_37), method = "cohort.count", 
                 conf.level = 0.95, units = 100, outcome = "as.columns")
test$massoc$RR.strata.wald

test <- epi.2by2(table(MOD3$ddsleepstart_Q4,MOD3$PreE), method = "cohort.count", 
                 conf.level = 0.95, units = 100, outcome = "as.columns")
test$massoc$RR.strata.wald

test <- epi.2by2(table(MOD3$ddsleepstart_Q4,MOD3$PreE_37), method = "cohort.count", 
                 conf.level = 0.95, units = 100, outcome = "as.columns")
test$massoc$RR.strata.wald


table(MOD2$ddsleepstart_low,MOD2$preeclampsia)
table(MOD2$ddsleepstart_quarter,MOD2$preeclampsia)

MOD2$PreE37 <- 0
MOD2$PreE37[which(MOD2$GAW<37 & MOD2$preeclampsia==1)] <- 1
table(MOD3$PreE37)
table(MOD3$ddsleepstart_quarter,MOD3$PreE37)

#################################################



MOD3 <- MOD2[is.na(MOD2$preeclampsia)==F,]

simresult200 <- logisticRR(PTB34 ~ ddsleepstart_low, data = MOD2, boot = TRUE, n.boot = 100)
simresult200$RR

simresult200 <- logisticRR(PTB34 ~ ddsleepstart_low, data = MOD2)
simresult200$RR

###############################################
#logistic
library(lme4)
library(sjmisc)

MOD2$PTB34 <- 0; MOD2$PTB34[which(MOD2$GAW<34)] <- 1
table(MOD2$PTB34)
MOD2$PTB37 <- 0; MOD2$PTB37[which(MOD2$GAW<37)] <- 1
table(MOD2$PTB37)

MOD2$ddsleepstart_high <- 0
MOD2$ddsleepstart_high[which(MOD2$ddsleepstart_low==0)] <- 1
table(MOD2$ddsleepstart_low)
table(MOD2$ddsleepstart_high)

MOD2$ddsleepstart_Q123 <- 0
MOD2$ddsleepstart_Q123[which(MOD2$ddsleepstart_Q4==0)] <- 1
table(MOD2$ddsleepstart_Q123)
table(MOD2$ddsleepstart_Q4)

MOD2$ddsleepstart_quarter <- ordered(as.numeric(as.character(MOD2$ddsleepstart_quarter)))

mylog <- glm(PTB37~ddsleepstart_quarter,MOD2,family = binomial(link = 'log'))
exp(confint(mylog))
exp(mylog$coefficients)

mylog <- glm(PTB34~ddsleepstart_quarter,MOD2,family = binomial(link = 'log'))
exp(confint(mylog))
exp(mylog$coefficients)

mylog <- glm(PTB37~ddsleepstart_high,MOD2,family = binomial(link = 'log'))
exp(confint(mylog))
exp(mylog$coefficients)

mylog <- glm(PTB34~ddsleepstart_high,MOD2,family = binomial(link = 'log'))
exp(confint(mylog))
exp(mylog$coefficients)

mylog <- glm(PTB37~ddsleepstart_Q123,MOD2,family = binomial(link = 'log'))
exp(confint(mylog))
exp(mylog$coefficients)

mylog <- glm(PTB34~ddsleepstart_Q123,MOD2,family = binomial(link = 'log'))
exp(confint(mylog))
exp(mylog$coefficients)

#####
#PreE
#####

MOD3$PreER <- 0
MOD3$PreER[which(MOD3$preeclampsia==1)] <- 1
MOD3$PreE_37R <- 0
MOD3$PreE_37R[which(MOD3$GAW<37 & MOD3$preeclampsia==1)] <- 1
table(MOD3$PreER)
table(MOD3$PreE_37R)

mylog <- glm(PreER~ddsleepstart_high,MOD3,family = binomial(link = 'log'))
exp(confint(mylog))
exp(mylog$coefficients)
mylog <- glm(PreER~ddsleepstart_quarter,MOD3,family = binomial(link = 'log'))
exp(confint(mylog))
exp(mylog$coefficients)
mylog <- glm(PreER~ddsleepstart_Q123,MOD3,family = binomial(link = 'log'))
exp(confint(mylog))
exp(mylog$coefficients)

mylog <- glm(PreE_37R~ddsleepstart_high,MOD3,family = binomial(link = 'log'))
exp(confint(mylog))
exp(mylog$coefficients)
mylog <- glm(PreE_37R~ddsleepstart_quarter,MOD3,family = binomial(link = 'log'))
exp(confint(mylog))
exp(mylog$coefficients)
mylog <- glm(PreE_37R~ddsleepstart_Q123,MOD3,family = binomial(link = 'log'))
exp(confint(mylog))
exp(mylog$coefficients)


###############################
write.csv(MOD2,'Chronotype/MOD2.csv',row.names = F)
write.csv(MOD3,'Chronotype/MOD3.csv',row.names = F)





































mylogit <- glm(PTB37~ddsleepstart_low,MOD2,family = 'binomial')
confint(mylogit)

mylogit <- glm(PTB37~ddsleepstart_Q4,MOD2,family = 'binomial')
confint(mylogit)















table(MOD2$race)
table(MOD2$education)
table(MOD2$marital)
table(MOD2$employed)
table(MOD2$nightshift_1trim)
table(MOD2$nightshift_2trim)
table(MOD2$nightshift_3trim)
table(MOD2$insur)

length(which(MOD2$GAW>=37))
length(which(MOD2$GAW<37 & MOD2$GAW>=34))
length(which(MOD2$GAW<34))

table(MOD2$smoke)
table(MOD2$alcohol)
table(MOD2$adi1)






















#who have all 3 trimesters?
MedianTable[,1:2]
complete_list <- c()
for(i in unique(MedianTable$ID)){
  if(1 %in% MedianTable$Trim[MedianTable$ID==i] &
     2 %in% MedianTable$Trim[MedianTable$ID==i] &
     3 %in% MedianTable$Trim[MedianTable$ID==i])complete_list <- c(complete_list,i)
}
CompleteTable <- MedianTable[MedianTable$ID %in% complete_list,]
length(complete_list)
#108 patients who have all trimesters

dim(CompleteTable)

boxplot(ddsleepstart~Trim,data=CompleteTable)
boxplot(ddsleepend~Trim,data=CompleteTable)
boxplot(sleep_start~Trim,data=CompleteTable)
boxplot(sleepend~Trim,data=CompleteTable)














widetable <- data.frame()
for(i in 1:134){
  templine <- c()
  for(j in 0:2){
    templine <- c(templine,MedianTable[i+134*j,3:13])
  }
  widetable <- rbind(widetable,templine)
  colnames(widetable) <- c()
}
widetable$ID <- complete_list

colnames(widetable) <- c("Onset1","Offset1","Mid1","Duration1","TotalAct1","SleepAct1","SleepFreq1","DDOnset1","DDOffset1","DDMid1","DDDuration1",
                         "Onset2","Offset2","Mid2","Duration2","TotalAct2","SleepAct2","SleepFreq2","DDOnset2","DDOffset2","DDMid2","DDDuration2",
                         "Onset3","Offset3","Mid3","Duration3","TotalAct3","SleepAct3","SleepFreq3","DDOnset3","DDOffset3","DDMid3","DDDuration3","ID")

boxplot(widetable[,8]-widetable[,8],widetable[,19]-widetable[,8],widetable[,30]-widetable[,8],outline = F)
wilcox.test(widetable[,8]-widetable[,8],widetable[,19]-widetable[,8])
wilcox.test(widetable[,8]-widetable[,8],widetable[,30]-widetable[,8])
wilcox.test(widetable[,19]-widetable[,8],widetable[,30]-widetable[,8])

boxplot(widetable[,9]-widetable[,9],widetable[,20]-widetable[,9],widetable[,31]-widetable[,9],outline = F)
wilcox.test(widetable[,9]-widetable[,9],widetable[,20]-widetable[,9])
wilcox.test(widetable[,9]-widetable[,9],widetable[,31]-widetable[,9])
wilcox.test(widetable[,20]-widetable[,9],widetable[,31]-widetable[,9])













Survdata <- data.frame()
for(i in unique(MedianTable$ID)){
  temp <- MedianTable[MedianTable$ID==i,c(3:13,18)]
  Survdata <- rbind(Survdata,c(i,apply(temp,2,mean)))
}
dim(Survdata)
colnames(Survdata) <- c('ID',colnames(MedianTable)[c(3:13,18)])
names(Survdata)
Survdata$ID <- as.character(Survdata$ID)

Survdata <- Survdata[which(Survdata$GA_Days > 0 ),]
dim(Survdata)
# 548 patients with delivery outcome

par(mfrow=c(3,4),mar=c(2,2,2,2))
for(i in c(2:5,9:12,6:8)){
  plot(Survdata[,i],Survdata$GA_Days,
       main=colnames(Survdata)[i],pch=19)
}

library(survival)
par(mfrow=c(3,4),mar=c(2,2,2,2))
for(i in c(2:5,9:12,6:8)){
  Survdata$GA_Weeks <- Survdata$GA_Days/7
  Survdata$low <- ifelse(Survdata[,i] <= median(Survdata[,i]),1,0)
  km_MOD <- survfit(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ low, data=Survdata)
  sdf <- survdiff(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ low, data=Survdata)
  p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
  plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
       xlim=c(100/7,290/7),col = c('red','blue'))
  text(100/7,0.9,'low',col='blue',pos=4)
  text(100/7,0.8,'high',col='red',pos=4)
  text(100/7,0.7,paste('P-value',p3),pos=4)
}

names(Survdata)
i <- 9
summary(Survdata[,i])
Survdata$GA_Weeks <- Survdata$GA_Days/7
Survdata$low <- ifelse(Survdata[,i] <= median(Survdata[,i]),1,0)
table(Survdata$low,ifelse(Survdata$GA_Weeks<37,1,0))
fisher.test(Survdata$low,ifelse(Survdata$GA_Weeks<37,1,0))

Survdata$quartile <- 3
Survdata$quartile[which(Survdata[,i]<quantile(Survdata[,i],0.75))] <- 2
Survdata$quartile[which(Survdata[,i]<quantile(Survdata[,i],0.5))] <- 1
Survdata$quartile[which(Survdata[,i]<quantile(Survdata[,i],0.25))] <- 0
table(Survdata$quartile)

Survdata$PTB <- ifelse(Survdata$GA_Weeks<37,1,0)

table(Survdata$q4,ifelse(Survdata$GA_Weeks<37,1,0))
fisher.test(Survdata$q4,ifelse(Survdata$GA_Weeks<37,1,0))
model <- glm(PTB ~ q4,family=binomial(link='log'),data=Survdata)
exp(coef(summary(model)))
exp(confint(model))

table(Survdata$low,ifelse(Survdata$GA_Weeks<37,1,0))
fisher.test(Survdata$low,ifelse(Survdata$GA_Weeks<37,1,0))
model <- glm(PTB ~ low,family=binomial(link='log'),data=Survdata)
exp(coef(summary(model)))
exp(confint(model))

Acti_list <- unique(Survdata$ID[which(Survdata$ddsleepstart >= quantile(Survdata$ddsleepstart,0.75))])

write.csv(Acti_list,'Chronotype/Irregular_PID.csv')


#32 weeks
table(Survdata$q4,ifelse(Survdata$GA_Weeks<32,1,0))
fisher.test(Survdata$q4,ifelse(Survdata$GA_Weeks<32,1,0))
model <- glm(PTB ~ q4,family=binomial(link='log'),data=Survdata)
exp(coef(summary(model)))
exp(confint(model))

table(Survdata$low,ifelse(Survdata$GA_Weeks<32,1,0))
fisher.test(Survdata$low,ifelse(Survdata$GA_Weeks<32,1,0))
model <- glm(PTB ~ low,family=binomial(link='log'),data=Survdata)
exp(coef(summary(model)))
exp(confint(model))





Survdata$q4 <- 1
Survdata$q4[which(Survdata[,i]<quantile(Survdata[,i],0.75))] <- 0
table(Survdata$q4)
table(Survdata$q4,ifelse(Survdata$GA_Weeks<37,1,0))
fisher.test(Survdata$q4,ifelse(Survdata$GA_Weeks<37,1,0))

i <- 9
Survdata$tercile <- 1
Survdata$tercile[which(Survdata[,i]<quantile(Survdata[,i],0.67))] <- 0
table(Survdata$tercile)
table(Survdata$tercile,ifelse(Survdata$GA_Weeks<37,1,0))
fisher.test(Survdata$tercile,ifelse(Survdata$GA_Weeks<37,1,0))


km_MOD <- survfit(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ low, data=Survdata)
sdf <- survdiff(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ low, data=Survdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,290/7),col = c('red','blue'))
text(100/7,0.9,'low',col='blue',pos=4)
text(100/7,0.8,'high',col='red',pos=4)
text(100/7,0.7,paste('P-value:',p3),pos=4)

colors <- rainbow(n=4,start=0.65,end=0.9)
km_MOD <- survfit(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ quartile, data=Survdata)
sdf <- survdiff(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ quartile, data=Survdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,290/7),col = colors)
text(100/7,0.9,'Q1',col=colors[1],pos=4)
text(100/7,0.8,'Q2',col=colors[2],pos=4)
text(100/7,0.7,'Q3',col=colors[3],pos=4)
text(100/7,0.6,'Q4',col=colors[4],pos=4)
text(100/7,0.5,paste('P-value:',p3),pos=4)

km_MOD <- survfit(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ q4, data=Survdata)
sdf <- survdiff(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ q4, data=Survdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,290/7),col = c('blue','red'))
text(100/7,0.9,'Q1~Q3',col='blue',pos=4)
text(100/7,0.8,'Q4',col='red',pos=4)
text(100/7,0.7,paste('P-value:',p3),pos=4)

Newlist <- Survdata$ID[which(Survdata$q4==1 & Survdata$GA_Weeks<37)]

colnames(Survdata)[1] <- 'record_id'
Survdata <- merge(Survdata,Indications[,c(1,3,6)],all.x = T)
Survdata$sptb <- Survdata$iptb <- 0
Survdata$sptb[which(Survdata$sptb_37wks==1)] <- 1
Survdata$iptb[which(Survdata$ptb_37wks==1 & Survdata$sptb_37wks==0)] <- 1



i <- 9
Surv1 <- Survdata[which(Survdata$q4==1),]
Surv2 <- Survdata[which(Survdata$q4==0),]
km_MOD <- survfit(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ q4, data=Survdata)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,295/7),lty=2)
for(i in 1:nrow(Surv1)){
  if(is.na(Surv1$GA_Weeks[i])==T | Surv1$GA_Weeks[i]>37)next
  points(Surv1$GA_Weeks[i],length(which(Surv1$GA_Weeks>Surv1$GA_Weeks[i]))/nrow(Surv1),
         pch=ifelse(Surv1$sptb[i]==1,20,20),
         col=ifelse(Surv1$sptb[i]==1,'red','green'))
}
for(i in 1:nrow(Surv2)){
  if(is.na(Surv2$GA_Weeks[i])==T | Surv2$GA_Weeks[i]>37)next
  points(Surv2$GA_Weeks[i],length(which(Surv2$GA_Weeks>Surv2$GA_Weeks[i]))/nrow(Surv2),
         pch=ifelse(Surv2$sptb[i]==1,20,20),
         col=ifelse(Surv2$sptb[i]==1,'red','green'))
}
points(15, 0.8,pch=20,col='green'); text(16,0.8,'induced',pos=4)
points(15, 0.75,pch=20,col='red'); text(16,0.75,'Spontaneous',pos=4)
text(36,0.55,'Irregular');text(41,0.55,'Rregular')
segments(35,0.7,35,1,lty=2);text(35.1,1,'35 Weeks',pos=4)

#######
#tercile
i <- 9
km_MOD <- survfit(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ tercile, data=Survdata)
Surv1 <- Survdata[which(Survdata$tercile==1),]
Surv2 <- Survdata[which(Survdata$tercile==0),]
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,295/7),lty=2)
for(i in 1:nrow(Surv1)){
  if(is.na(Surv1$GA_Weeks[i])==T | Surv1$GA_Weeks[i]>37)next
  points(Surv1$GA_Weeks[i],length(which(Surv1$GA_Weeks>Surv1$GA_Weeks[i]))/nrow(Surv1),
         pch=ifelse(Surv1$sptb[i]==1,20,20),
         col=ifelse(Surv1$sptb[i]==1,'red','green'))
}
for(i in 1:nrow(Surv2)){
  if(is.na(Surv2$GA_Weeks[i])==T | Surv2$GA_Weeks[i]>37)next
  points(Surv2$GA_Weeks[i],length(which(Surv2$GA_Weeks>Surv2$GA_Weeks[i]))/nrow(Surv2),
         pch=ifelse(Surv2$sptb[i]==1,20,20),
         col=ifelse(Surv2$sptb[i]==1,'red','green'))
}
points(15, 0.8,pch=20,col='green'); text(16,0.8,'induced',pos=4)
points(15, 0.75,pch=20,col='red'); text(16,0.75,'Spontaneous',pos=4)
text(36,0.55,'Irregular');text(41,0.55,'Rregular')
segments(35,0.7,35,1,lty=2);text(35.1,1,'35 Weeks',pos=4)

#q4 and tercile, which is better
length(which(Survdata$q4==1 & Survdata$PTB==1))/length(which(Survdata$q4==1))
length(which(Survdata$tercile==1 & Survdata$PTB==1))/length(which(Survdata$tercile==1))
#q4 is better

#######Indications
Survdata$Membrane <- 0
Survdata$Membrane[which(Survdata$record_id %in% Memb_list)] <- 1
table(Survdata$Membrane)

Survdata$MF <- 0
Survdata$MF[which(Survdata$record_id %in% MF_list)] <- 1
table(Survdata$MF)

Survdata$Dila <- 0
Survdata$Dila[which(Survdata$record_id %in% Dila_list)] <- 1
table(Survdata$Dila)

i <- 9
Surv1 <- Survdata[which(Survdata$q4==1),]
Surv2 <- Survdata[which(Survdata$q4==0),]
km_MOD <- survfit(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ q4, data=Survdata)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,295/7),lty=1,lwd=0.6)
for(i in 1:nrow(Surv1)){
  if(is.na(Surv1$GA_Weeks[i])==T | Surv1$GA_Weeks[i]>37)next
  points(Surv1$GA_Weeks[i],length(which(Surv1$GA_Weeks>Surv1$GA_Weeks[i]))/nrow(Surv1),
         cex=ifelse(Surv1$MF[i]==1,1.1,0),pch=3,col='red')
}
for(i in 1:nrow(Surv1)){
  if(is.na(Surv1$GA_Weeks[i])==T | Surv1$GA_Weeks[i]>37)next
  points(Surv1$GA_Weeks[i],length(which(Surv1$GA_Weeks>Surv1$GA_Weeks[i]))/nrow(Surv1),
         cex=ifelse(Surv1$Dila[i]==1,1.1,0),pch=4,col='green')
}

for(i in 1:nrow(Surv2)){
  if(is.na(Surv2$GA_Weeks[i])==T | Surv2$GA_Weeks[i]>37)next
  points(Surv2$GA_Weeks[i],length(which(Surv2$GA_Weeks>Surv2$GA_Weeks[i]))/nrow(Surv2),
         cex=ifelse(Surv2$MF[i]==1,1.1,0),pch=3,col='red')
}
for(i in 1:nrow(Surv2)){
  if(is.na(Surv2$GA_Weeks[i])==T | Surv2$GA_Weeks[i]>37)next
  points(Surv2$GA_Weeks[i],length(which(Surv2$GA_Weeks>Surv2$GA_Weeks[i]))/nrow(Surv2),
         cex=ifelse(Surv2$Dila[i]==1,1.1,0),pch=4,col='green')
}

points(15, 0.8,pch=4,col='green',cex=1.1); text(16,0.8,'Dilation',pos=4)
points(15, 0.75,pch=3,col='red',cex=1.1); text(16,0.75,'Iatrogenic',pos=4)
text(36,0.55,'Irregular');text(41,0.55,'Regular')
segments(37,0.7,37,1,lty=2);text(37.1,1,'37 Weeks',pos=4)


# What about cervix index
dim(rfdata)
names(rfdata)
rfdata$GAW <- rfdata$GA/7
# define indications
rfdata$Membrane <- 0
rfdata$Membrane[which(rfdata$record_id %in% Memb_list)] <- 1
table(rfdata$Membrane)

rfdata$MF <- 0
rfdata$MF[which(rfdata$record_id %in% MF_list)] <- 1
table(rfdata$MF)

rfdata$Dila <- 0
rfdata$Dila[which(rfdata$record_id %in% Dila_list)] <- 1
table(rfdata$Dila)
#########################
km_MOD <- survfit(Surv(GAW, rep(1,nrow(rfdata))) ~ new, data=rfdata)
sdf <- survdiff(Surv(GAW, rep(1,nrow(rfdata))) ~ new, data=rfdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = 'Cervix Risk Index',
     xlim=c(100/7,295/7),lty=1,col=c('blue','red'))
text(100/7,0.9,paste('P-value=',p3),pos=4)

#indications
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = 'Cervix Risk Index',
     xlim=c(100/7,295/7),lty=1,lwd=0.6)
Surv1 <- rfdata[which(rfdata$new==1 & is.na(rfdata$GAW)==F),]
Surv2 <- rfdata[which(rfdata$new==0 & is.na(rfdata$GAW)==F),]
for(i in 1:nrow(Surv1)){
  if(is.na(Surv1$GAW[i])==T | Surv1$GAW[i]>37)next
  points(Surv1$GAW[i],length(which(Surv1$GAW>Surv1$GAW[i]))/nrow(Surv1),
         cex=ifelse(Surv1$MF[i]==1,1.1,0),pch=3,col='red')
}
for(i in 1:nrow(Surv1)){
  if(is.na(Surv1$GAW[i])==T | Surv1$GAW[i]>37)next
  points(Surv1$GAW[i],length(which(Surv1$GAW>Surv1$GAW[i]))/nrow(Surv1),
         cex=ifelse(Surv1$Dila[i]==1,1.1,0),pch=4,col='green')
}

for(i in 1:nrow(Surv2)){
  if(is.na(Surv2$GAW[i])==T | Surv2$GAW[i]>37)next
  points(Surv2$GAW[i],length(which(Surv2$GAW>Surv2$GAW[i]))/nrow(Surv2),
         cex=ifelse(Surv2$MF[i]==1,1.1,0),pch=3,col='red')
}
for(i in 1:nrow(Surv2)){
  if(is.na(Surv2$GAW[i])==T | Surv2$GAW[i]>37)next
  points(Surv2$GAW[i],length(which(Surv2$GAW>Surv2$GAW[i]))/nrow(Surv2),
         cex=ifelse(Surv2$Dila[i]==1,1.1,0),pch=4,col='green')
}

#pie chart
# slices <- c(10, 12, 4, 16, 8) 
# lbls <- c("US", "UK", "Australia", "Germany", "France")
# pct <- round(slices/sum(slices)*100)
# lbls <- paste(lbls, pct) # add percents to labels 
# lbls <- paste(lbls,"%",sep="") # ad % to labels 
# pie(slices,labels = lbls, col=rainbow(length(lbls)),
#     main="Pie Chart of Countries")
length(which(Surv1$ptb_37wks==1))/nrow(Surv1)
#ptb ratio= 17.9%
length(which(Surv1$ptb_37wks==1 & Surv1$Dila==1))/nrow(Surv1)
#Dilation 4.5%
length(which(Surv1$ptb_37wks==1 & Surv1$MF==1))/nrow(Surv1)
#Preeclamsia and Fetal: 6.16%
length(which(Surv1$ptb_37wks==1 & Surv1$Membrane==1))/nrow(Surv1)
#Membrane: 7.28%

length(which(Surv2$ptb_37wks==1))/nrow(Surv2)
#ptb ratio= 13.1%
length(which(Surv2$ptb_37wks==1 & Surv2$Dila==1))/nrow(Surv2)
#Dilation 2.6%
length(which(Surv2$ptb_37wks==1 & Surv2$MF==1))/nrow(Surv2)
#Preeclamsia and Fetal: 6.26%
length(which(Surv2$ptb_37wks==1 & Surv2$Membrane==1))/nrow(Surv2)
#Membrane: 3.97%

#Pie Chart of activity 
length(which(Surv1$ptb_37wks==1))/nrow(Surv1)
#ptb ratio= 21.17%
length(which(Surv1$ptb_37wks==1 & Surv1$Dila==1))/length(which(Surv1$ptb_37wks==1))
#Dilation 24.14%
length(which(Surv1$ptb_37wks==1 & Surv1$MF==1))/length(which(Surv1$ptb_37wks==1))
#Preeclamsia and Fetal: 51.72%
length(which(Surv1$ptb_37wks==1 & Surv1$Membrane==1))/length(which(Surv1$ptb_37wks==1))
#Membrane: 27.59%

length(which(Surv2$ptb_37wks==1))/nrow(Surv2)
#ptb ratio= 10.95%
length(which(Surv2$ptb_37wks==1 & Surv2$Dila==1))/length(which(Surv2$ptb_37wks==1))
#Dilation 17.78%
length(which(Surv2$ptb_37wks==1 & Surv2$MF==1))/length(which(Surv2$ptb_37wks==1))
#Preeclamsia and Fetal: 33.33%
length(which(Surv2$ptb_37wks==1 & Surv2$Membrane==1))/length(which(Surv2$ptb_37wks==1))
#Membrane: 44.44%

#Pie Chart of activity 
length(which(Surv1$ptb_37wks==1))/nrow(Surv1)
#ptb ratio= 21.17%
length(which(Surv1$ptb_37wks==1 & Surv1$Dila==1))/nrow(Surv1)
#Dilation 24.14%
length(which(Surv1$ptb_37wks==1 & Surv1$MF==1))/nrow(Surv1)
#Preeclamsia and Fetal: 51.72%
length(which(Surv1$ptb_37wks==1 & Surv1$Membrane==1))/nrow(Surv1)
#Membrane: 27.59%

length(which(Surv2$ptb_37wks==1))/nrow(Surv2)
#ptb ratio= 10.95%
length(which(Surv2$ptb_37wks==1 & Surv2$Dila==1))/nrow(Surv2)
#Dilation 17.78%
length(which(Surv2$ptb_37wks==1 & Surv2$MF==1))/nrow(Surv2)
#Preeclamsia and Fetal: 33.33%
length(which(Surv2$ptb_37wks==1 & Surv2$Membrane==1))/nrow(Surv2)
#Membrane: 44.44%

par(mfrow=c(1,2))
slices <- c(7, 15, 8)
lbls <- c("Dilation", "Iatrogenic", "Membrane")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels
pie(slices,labels = lbls, col=rainbow(n=3,start=0.65,end=0.9))

slices <- c(8, 15, 20)
lbls <- c("Dilation", "Iatrogenic", "Membrane")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels
pie(slices,labels = lbls, col=rainbow(n=3,start=0.65,end=0.9))













#censor at 37 weeks
km_MOD <- survfit(Surv(GA_Weeks, ifelse(Survdata$GA_Weeks>37,0,1)) ~ low, data=Survdata)
sdf <- survdiff(Surv(GA_Weeks, ifelse(Survdata$GA_Weeks>37,0,1)) ~ low, data=Survdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,290/7),col = c('red','blue'))
abline(v=37,lty=2)
text(100/7,0.9,'low',col='blue',pos=4)
text(100/7,0.8,'high',col='red',pos=4)
text(100/7,0.7,paste('P-value:',p3),pos=4)

colors <- rainbow(n=4,start=0.65,end=0.9)
km_MOD <- survfit(Surv(GA_Weeks, ifelse(Survdata$GA_Weeks>37,0,1)) ~ quartile, data=Survdata)
sdf <- survdiff(Surv(GA_Weeks, ifelse(Survdata$GA_Weeks>37,0,1)) ~ quartile, data=Survdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,290/7),col = colors)
abline(v=37,lty=2)
text(100/7,0.9,'Q1',col=colors[1],pos=4)
text(100/7,0.8,'Q2',col=colors[2],pos=4)
text(100/7,0.7,'Q3',col=colors[3],pos=4)
text(100/7,0.6,'Q4',col=colors[4],pos=4)
text(100/7,0.5,paste('P-value:',p3),pos=4)

km_MOD <- survfit(Surv(GA_Weeks, ifelse(Survdata$GA_Weeks>37,0,1)) ~ q4, data=Survdata)
sdf <- survdiff(Surv(GA_Weeks, ifelse(Survdata$GA_Weeks>37,0,1)) ~ q4, data=Survdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,290/7),col = c('blue','red'))
abline(v=37,lty=2)
text(100/7,0.9,'Q1~Q3',col='blue',pos=4)
text(100/7,0.8,'Q4',col='red',pos=4)
text(100/7,0.7,paste('P-value:',p3),pos=4)


# activity frequency
i <- 8
Survdata$GA_Weeks <- Survdata$GA_Days/7
Survdata$low <- ifelse(Survdata[,i] <= median(Survdata[,i]),1,0)

Survdata$quartile <- 3
Survdata$quartile[which(Survdata[,i]<quantile(Survdata[,i],0.75))] <- 2
Survdata$quartile[which(Survdata[,i]<quantile(Survdata[,i],0.5))] <- 1
Survdata$quartile[which(Survdata[,i]<quantile(Survdata[,i],0.25))] <- 0
table(Survdata$quartile)

Survdata$q4 <- 1
Survdata$q4[which(Survdata[,i]<quantile(Survdata[,i],0.75))] <- 0
table(Survdata$q4)

names(Survdata)
Survdata$status <- 1
Survdata$SurvObj <- with(Survdata, Surv(GA_Weeks, status == 1))
cox.all <- coxph(SurvObj ~ ., data = Survdata[,c(17,20)])
summary(cox.all)


km_MOD <- survfit(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ low, data=Survdata)
sdf <- survdiff(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ low, data=Survdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,290/7),col = c('red','blue'))
text(100/7,0.9,'low',col='blue',pos=4)
text(100/7,0.8,'high',col='red',pos=4)
text(100/7,0.7,paste('P-value:',p3),pos=4)

colors <- rainbow(n=4,start=0.65,end=0.9)
km_MOD <- survfit(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ quartile, data=Survdata)
sdf <- survdiff(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ quartile, data=Survdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,290/7),col = colors)
text(100/7,0.9,'Q1',col=colors[1],pos=4)
text(100/7,0.8,'Q2',col=colors[2],pos=4)
text(100/7,0.7,'Q3',col=colors[3],pos=4)
text(100/7,0.6,'Q4',col=colors[4],pos=4)
text(100/7,0.5,paste('P-value:',p3),pos=4)

km_MOD <- survfit(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ q4, data=Survdata)
sdf <- survdiff(Surv(GA_Weeks, rep(1,nrow(Survdata))) ~ q4, data=Survdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,290/7),col = c('blue','red'))
text(100/7,0.9,'Q1~Q3',col='blue',pos=4)
text(100/7,0.8,'Q4',col='red',pos=4)
text(100/7,0.7,paste('P-value:',p3),pos=4)

#censor at 37 weeks
km_MOD <- survfit(Surv(GA_Weeks, ifelse(Survdata$GA_Weeks>37,0,1)) ~ low, data=Survdata)
sdf <- survdiff(Surv(GA_Weeks, ifelse(Survdata$GA_Weeks>37,0,1)) ~ low, data=Survdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,290/7),col = c('red','blue'))
abline(v=37,lty=2)
text(100/7,0.9,'low',col='blue',pos=4)
text(100/7,0.8,'high',col='red',pos=4)
text(100/7,0.7,paste('P-value:',p3),pos=4)

colors <- rainbow(n=4,start=0.65,end=0.9)
km_MOD <- survfit(Surv(GA_Weeks, ifelse(Survdata$GA_Weeks>37,0,1)) ~ quartile, data=Survdata)
sdf <- survdiff(Surv(GA_Weeks, ifelse(Survdata$GA_Weeks>37,0,1)) ~ quartile, data=Survdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,290/7),col = colors)
abline(v=37,lty=2)
text(100/7,0.9,'Q1',col=colors[1],pos=4)
text(100/7,0.8,'Q2',col=colors[2],pos=4)
text(100/7,0.7,'Q3',col=colors[3],pos=4)
text(100/7,0.6,'Q4',col=colors[4],pos=4)
text(100/7,0.5,paste('P-value:',p3),pos=4)

km_MOD <- survfit(Surv(GA_Weeks, ifelse(Survdata$GA_Weeks>37,0,1)) ~ q4, data=Survdata)
sdf <- survdiff(Surv(GA_Weeks, ifelse(Survdata$GA_Weeks>37,0,1)) ~ q4, data=Survdata)
p3 <- round(1 - pchisq(sdf$chisq, length(sdf$n) - 1),5)
plot(km_MOD, xlab="Gestatonal Age (Weeks)", main = colnames(Survdata)[i],
     xlim=c(100/7,290/7),col = c('blue','red'))
abline(v=37,lty=2)
text(100/7,0.9,'Q1~Q3',col='blue',pos=4)
text(100/7,0.8,'Q4',col='red',pos=4)
text(100/7,0.7,paste('P-value:',p3),pos=4)



# Most irregular activity patient list
hist(Survdata$ddsleepstart)
Survdata$record_id[which(Survdata$ddsleepstart>100)]

#############Function
odds_to_rr <- function(fit) {
  # check model family
  fitinfo <- get_glm_family(fit)
  
  # no binomial model with logit-link?
  if (!fitinfo$is_bin && !fitinfo$is_logit)
    stop("`fit` must be a binomial model with logit-link (logistic regression).", call. = F)
  
  # get model estimates
  est <- insight::get_parameters(fit)
  est[[2]] <- exp(est[[2]])
  
  # get confidence intervals
  if (is_merMod(fit))
    ci <- stats::confint(fit, method = "Wald", parm = "beta_")
  else
    ci <- stats::confint(fit)
  
  # bind to data frame
  or.dat <- data.frame(est, exp(ci))
  colnames(or.dat) <- c("Parameter", "OR", "CI_low", "CI_high")
  
  # get P0, i.e. the incidence ratio of the outcome for the
  # non-exposed group
  modfram <- insight::get_data(fit)
  
  # make sure that outcome is 0/1-numeric, so we can simply
  # compute the mean to get the ratio
  outcome <- sjmisc::recode_to(sjlabelled::as_numeric(insight::get_response(fit)))
  
  P0 <- c()
  for (i in 1:nrow(est)) {
    P0 <- c(P0, .baseline_risk_for_predictor(modfram, outcome, est[[1]][i]))
  }
  
  # compute relative risks for estimate and confidence intervals
  rr.dat <- or.dat[, 2:4] / ((1 - P0) + (P0 * or.dat[, 2:4]))
  rr.dat <- cbind(or.dat$Parameter, or.dat$OR, rr.dat)
  
  colnames(rr.dat) <- c("Parameter", "Odds Ratio", "Risk Ratio", "CI_low", "CI_high")
  rownames(rr.dat) <- NULL
  
  rr.dat
}


.baseline_risk_for_predictor <- function(data, outcome, parameter) {
  if (parameter == "(Intercept)") return(mean(outcome))
  
  if (!(parameter %in% colnames(data))) {
    find.factors <- lapply(colnames(data), function(.i) {
      v <- data[[.i]]
      if (is.factor(v)) {
        return(paste0(.i, levels(v)))
      }
      return(.i)
    })
    names(find.factors) <- colnames(data)
    parameter <- names(find.factors)[which(sapply(find.factors, function(.i) {
      parameter %in% .i
    }))]
  }
  
  if (is.numeric(data[[parameter]])) {
    mean(outcome)
  } else {
    p <- prop.table(table(data[[parameter]], outcome))
    p[1, 2] / sum(p[1, ])
  }
}


#' @rdname odds_to_rr
#' @export
or_to_rr <- function(or, p0) {
  or / (1 - p0 + (p0 * or))
}





