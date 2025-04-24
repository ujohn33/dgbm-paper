
## set language setting
Sys.setlocale(locale = "en_US.utf8") ## English US Linux
Sys.setlocale(locale = "en_US") ## English US Mac
Sys.setlocale(locale = "English_United States.1252") ## English US Windows

suppressMessages({
    library(data.table)
    library(RcppRoll)
    library(dplyr)
    library(ragg)
    #library(lightgbm)
})

# Garbage collection
igc <- function() {
    invisible(gc()); invisible(gc())   
}


dm = 1941 #? +28
H=1#1#28

path <- "/scratch/brussel/105/vsc10528/LSSboost/data/m5-forecasting-accuracy/"

calendar <- fread(file.path(path, "calendar.csv"))
selling_prices <- fread(file.path(path, "sell_prices.csv"))
#sample_submission <- fread(file.path(path, "sample_submission.csv"))
#sales <- fread(file.path(path, "sales_train_validation.csv"))
sales <- fread(file.path(path, "sales_train_evaluation.csv"))


# Calendar
calendar[['date']] <- as.Date(calendar[['date']])
calendar[['dayofmonth']] <- as.numeric(format(calendar[['date']], "%d"))
#calendar[, `:=`( weekday = NULL, 
#                d = as.integer(substring(d, 3)))]
cols <- c("event_name_1", "event_type_1")#, "event_name_2", "event_type_2")
calendar[, (cols) := lapply(.SD, function(z) as.integer(as.factor(z))), .SDcols = cols]

# Selling prices                   
selling_prices[, `:=`(
    sell_price_rel_diff = sell_price / dplyr::lag(sell_price) - 1,
    sell_price_cumrel = (sell_price - cummin(sell_price)) / (1 + cummax(sell_price) - cummin(sell_price)),
    sell_price_roll_sd7 = roll_sdr(sell_price, n = 7)
  ), by = c("store_id", "item_id")]

# Sales: Reshape
sales[, id := gsub("_validation", "", id)]                         
empty_dt = matrix(NA_integer_, ncol =  H, nrow = 1, dimnames = 
                  list(NULL, paste("d", dm + 1:( H), sep = "_")))

sales <- cbind(sales, empty_dt)
sales <- melt(sales, id.vars = c("id", "item_id", "dept_id", "cat_id", "store_id", "state_id"), 
              variable.name = "d", value = "demand")
sales[, d := as.integer(substring(d, 3))]

                            
# Sales: Reduce size
ptmax<- 730
sales <- sales[d >= dm-ptmax-375] #370 due to annual lags
calendar <- calendar[d >= dm-ptmax-375] #370 due to annual lags
gc()

n_item = 3049
n_stores = 10
n_all<- n_item*n_stores

n_dates<- 1941 #dim(train)[1]/n_all - 29 ## oos

iitem<- 1:n_item#30#12
iin<- length(iitem)#20
tmp<-as.factor(sales[,item_id])

iitem_id<- which(tmp  %in% tmp[1:n_item][iitem])

sales<- sales[iitem_id,]
remove(iitem_id)
##
sales[, 'demand_store_id' := mean(demand, na.rm=TRUE), by= c("store_id", "d") ]
sales[, 'demand_item_id' := mean(demand, na.rm=TRUE), by= c("item_id", "d") ]
sales[, 'demand_all_id' := mean(demand, na.rm=TRUE), by= c("d") ]
sales[1:30,c("demand", "item_id","store_id", "d", "demand_store_id", "demand_item_id", "demand_all_id")]

# Sales: Feature construction: Subset of features from very fst kernel
stopifnot(!is.unsorted(sales$d))
sales[, demand_lag_tH := dplyr::lag(demand, H), by = "id"]
sales[, demand_lag_tHp1 := dplyr::lag(demand, H+1), by = "id"]
#sales[, demand_lag_tHp2 := dplyr::lag(demand, H+2), by = "id"]
#sales[, demand_lag_tHp3 := dplyr::lag(demand, H+3), by = "id"]
#sales[, demand_lag_tHp4 := dplyr::lag(demand, H+4), by = "id"]
#sales[, demand_lag_tHp5 := dplyr::lag(demand, H+5), by = "id"]
sales[, demand_lag_tHp6 := dplyr::lag(demand, H+6), by = "id"]
sales[, demand_lag_ann7 := dplyr::lag(demand, 52*7), by = "id"]
sales[, `:=`(demand_rolling_mean_t7 = roll_meanr(demand_lag_tH, 7),
             demand_rolling_mean_t14 = roll_meanr(demand_lag_tH, 14),
             demand_rolling_mean_t28 = roll_meanr(demand_lag_tH, 28),
             demand_rolling_mean_t56 = roll_meanr(demand_lag_tH, 56),
             demand_rolling_mean_ann7 = roll_meanr(demand_lag_ann7, 7)
			), 
      by = "id"]

sales[, demand_store_id_lag_tH := dplyr::lag(demand_store_id, H), by = "id"]
sales[, demand_store_id_lag_tHp1 := dplyr::lag(demand_store_id, H+1), by = "id"]
#sales[, demand_store_id_lag_tHp2 := dplyr::lag(demand_store_id, H+2), by = "id"]
#sales[, demand_store_id_lag_tHp3 := dplyr::lag(demand_store_id, H+3), by = "id"]
#sales[, demand_store_id_lag_tHp4 := dplyr::lag(demand_store_id, H+4), by = "id"]
#sales[, demand_store_id_lag_tHp5 := dplyr::lag(demand_store_id, H+5), by = "id"]
sales[, demand_store_id_lag_tHp6 := dplyr::lag(demand_store_id, H+6), by = "id"]
sales[, demand_store_id_lag_ann7 := dplyr::lag(demand_store_id, 52*7), by = "id"]
sales[, `:=`(demand_store_id_rolling_mean_t7 = roll_meanr(demand_store_id_lag_tH, 7),
             demand_store_id_rolling_mean_t14 = roll_meanr(demand_store_id_lag_tH, 14),
             demand_store_id_rolling_mean_t28 = roll_meanr(demand_store_id_lag_tH, 28),
             demand_store_id_rolling_mean_t56 = roll_meanr(demand_store_id_lag_tH, 56),
             demand_store_id_rolling_mean_ann7 = roll_meanr(demand_store_id_lag_ann7, 7)
			), 
      by = "id"]

sales[, demand_item_id_lag_tH := dplyr::lag(demand_item_id, H), by = "id"]
sales[, demand_item_id_lag_tHp1 := dplyr::lag(demand_item_id, H+1), by = "id"]
#sales[, demand_item_id_lag_tHp2 := dplyr::lag(demand_item_id, H+2), by = "id"]
#sales[, demand_item_id_lag_tHp3 := dplyr::lag(demand_item_id, H+3), by = "id"]
#sales[, demand_item_id_lag_tHp4 := dplyr::lag(demand_item_id, H+4), by = "id"]
#sales[, demand_item_id_lag_tHp5 := dplyr::lag(demand_item_id, H+5), by = "id"]
sales[, demand_item_id_lag_tHp6 := dplyr::lag(demand_item_id, H+6), by = "id"]
sales[, demand_item_id_lag_ann7 := dplyr::lag(demand_item_id, 52*7), by = "id"]
sales[, `:=`(demand_item_id_rolling_mean_t7 = roll_meanr(demand_item_id_lag_tH, 7),
             demand_item_id_rolling_mean_t14 = roll_meanr(demand_item_id_lag_tH, 14),
             demand_item_id_rolling_mean_t28 = roll_meanr(demand_item_id_lag_tH, 28),
             demand_item_id_rolling_mean_t56 = roll_meanr(demand_item_id_lag_tH, 56),
             demand_item_id_rolling_mean_ann7 = roll_meanr(demand_item_id_lag_ann7, 7)
			), 
      by = "id"]

sales[, demand_all_id_lag_tH := dplyr::lag(demand_all_id, H), by = "id"]
sales[, demand_all_id_lag_tHp1 := dplyr::lag(demand_all_id, H+1), by = "id"]
#sales[, demand_all_id_lag_tHp2 := dplyr::lag(demand_all_id, H+2), by = "id"]
#sales[, demand_all_id_lag_tHp3 := dplyr::lag(demand_all_id, H+3), by = "id"]
#sales[, demand_all_id_lag_tHp4 := dplyr::lag(demand_all_id, H+4), by = "id"]
#sales[, demand_all_id_lag_tHp5 := dplyr::lag(demand_all_id, H+5), by = "id"]
sales[, demand_all_id_lag_tHp6 := dplyr::lag(demand_all_id, H+6), by = "id"]
sales[, demand_all_id_lag_ann7 := dplyr::lag(demand_all_id, 52*7), by = "id"]
sales[, `:=`(demand_all_id_rolling_mean_t7 = roll_meanr(demand_all_id_lag_tH, 7),
             demand_all_id_rolling_mean_t14 = roll_meanr(demand_all_id_lag_tH, 14),
             demand_all_id_rolling_mean_t28 = roll_meanr(demand_all_id_lag_tH, 28),
             demand_all_id_rolling_mean_t56 = roll_meanr(demand_all_id_lag_tH, 56),
             demand_all_id_rolling_mean_ann7 = roll_meanr(demand_all_id_lag_ann7, 7)
			), 
      by = "id"]



igc()
                            
#sales <- sales[d >= dm | !is.na(rolling_mean_ann28)]
sales <- sales[ !is.na(demand_rolling_mean_ann7) | !is.na(demand_rolling_mean_t28)]
#sales <- sales[d >= dm  | d]

#sales[ !is.na(rolling_mean_t364)]
gc()



## Merge calendar to sales
sales <- calendar[sales, on = "d"]
igc()
rm(calendar)
#sales['idn']<- sales[['id']]
sales[, 'idn' := id]


# Merge selling prices to sales and drop key
train <- selling_prices[sales, on = c('store_id', 'item_id', 'wm_yr_wk')][, wm_yr_wk := NULL]
rm(sales, selling_prices)
igc()

gc()

#(0:(dm-1)) *30491 +2

#ptmax<- 730
train <- train[d > dm - ptmax]

## REMOVE data between T+1 and T+H
train<- train[!(d<max(train[,d]) & d>max(train[,d])-H ),]

# Turn non-numerics to integer
cols <- c("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
train[, (cols) := lapply(.SD, function(z) as.integer(as.factor(z))), .SDcols = cols]

gc()

write.csv(train, "/scratch/brussel/105/vsc10528/LSSboost/data/m5-forecasting-accuracy/train_preprocessed.csv")

train <- fread("/scratch/brussel/105/vsc10528/LSSboost/data/m5-forecasting-accuracy/train_preprocessed.csv")
gc()

colnames(train)

#library(foreach)
library(parallel)
#library(changepoint)
library(glmnet)
library(pscl)
#registerDoParallel(8)
library(gamlss)
library(gamlss.add)
library(gamlss.dist)
library(gamlss.inf)
library(gamlss.lasso)





n_item = 3049
n_stores = 10
n_all<- n_item*n_stores
PRED<- array(, dim=c(n_item*n_stores,H,3))
PREDname<- array(, dim=c(n_item*n_stores,3))

n_dates<- ptmax#1941 #dim(train)[1]/n_all - 29 ## oos

set.seed(1234)
do_items = 1:n_item
do_items <-  sort( sample(1:n_item, 80) )

iitem<- 578
isto<- 10

ii<- 1
idx<- seq(ii, n_dates*n_all+ii, n_all)
#OUT<- as.data.frame(train[1:n_all, idn])

# names(out)
#OOnam<-  c( "mu","sigma" , "nu","tau","z0" ,"mu.tval","sigma.tval", "nu.tval","tau.tval", "z0.tval",    "mu.pval",    "sigma.pval","nu.pval","tau.pval", "z0.pval" ,   "d0"  , "d1" , "d2", "d3", "d4", "AIC",     "BIC")
#OM<- array(,dim=c(length(MODL),  length(DATlist), n_all, length(OOnam)))
#dimnames(OM)<- list(MODL, seq_along(DATlist), 1:n_all, OOnam)

#for(imod in 1:3){
#	aoo<- paste("out", MODL[imod], sep="")
#	assign( aoo, as.data.frame(matrix(,0,0)))
#}
OO<- as.data.frame(matrix(,0,0))

iiseq<-1:n_all
set.seed(1234)
#iiseq<- sort(sample(1:n_all, 100))
gc()
#DN<- 730# data used


idp<- which(train[,id] == train[1,id] ) -1
for(ii in iiseq){
	OO[ii,"item_id"]<- train[idp[1]+ii,c("item_id")]
	OO[ii,"store_id"]<- train[idp[1]+ii,c("store_id")]
	idx<- idp+ii #seq(ii, (n_dates-1)*n_all+ii, n_all)
	idxx<- idx[!is.na(train[idx,sell_price])]
	yy<- train[idxx, demand]
	idy<- !is.na(yy)
	yy<- yy[idy]
	yya<- train[idxx[idy], demand_rolling_mean_ann7]
	OO[ii,"n"]<- length(idxx)
	OO[ii,"min"]<- min(yy)
	OO[ii,"max"]<- max(yy)
  OO[ii,"mean"]<- mean(yy)
	OO[ii,"sd"]<- sqrt(var(yy))
	OO[ii,"z0"]<- mean(yy==0)
	OO[ii,"z1"]<- mean(yy==1)
	yacf<- acf(yy, lag.max=7, plot=FALSE)$acf[-1,,]
	ypacf<- pacf(yy, lag.max=7, plot=FALSE)$acf[,,]
  yacf[is.nan(yacf)]<- 0
	ypacf[is.nan(ypacf)]<- 0
	OO[ii,"acf1"]<- yacf[1]
	OO[ii,"acf7"]<- yacf[7]
	OO[ii,"pacf7"]<- ypacf[7]
	OO[ii,"acann7"]<- cor(yya,yy, use="pairwise.complete.obs")
	cat(ii)
}
OO[is.na(OO)]<- 0
save(OO, file = "/scratch/brussel/105/vsc10528/LSSboost/data/OO.RData")

load("/scratch/brussel/105/vsc10528/LSSboost/data/OO.RData")


OOO<- OO[1:n_item,]
for(i in 1:n_item) OOO[i,]<- apply(OO[i+n_item*(1:n_stores-1),],2,mean)
idi<- 1:n_item
XX<- cbind( OOO[idi, "n"], log(OOO[idi, "mean"]), log(OOO[idi, "sd"]), OOO[idi, "acf1"], OOO[idi, "pacf7"],  OOO[idi, "acann7"],  OOO[idi, "z0"])
XX[is.na(XX)]<- 0

#library(mclust)
#idi<- 1:200
#XX<- cbind( OO[idi, "n"], log(OO[idi, "mean"]), log(OO[idi, "sd"]), OO[idi, "acf1"], OO[idi, "acf7"] )
#cmod<- Mclust(XX)
#plot(cmod)


#idi<- 1:n_item
#XX<- cbind( OO[idi, "n"], log(OO[idi, "mean"]), log(OO[idi, "sd"]), OO[idi, "acf1"], OO[idi, "acf7"] )
#cmod<- Mclust(XX, G=1:20)
#plot(cmod)

#cmod2<- Mclust(XX, G=1:40, "EEE")
#plot(cmod2)

set.seed(1234)
mx<- 100
cmod<- kmeans(scale(XX), mx, nstart=100, iter.max = 200)

morder<- c(1,6,3,5,7,2,4,8,9)
COL<- c(1, 2:8, "orange")

#pdf("/scratch/brussel/105/vsc10528/LSSboost/data/cluster.pdf", width=12, height=10)

# Replaces pdf(...) with a headless-friendly PNG output
agg_png("/scratch/brussel/105/vsc10528/LSSboost/data/cluster.png", width = 1200, height = 1000, res = 150)
par(family="Times", mar=c(0.04,0.04,0.04,0.04), oma=c(0,0,0,0))

gsel<- c("n", "log mean", "log sd", "acf lag=1", "pacf lag=7", "ann. cor", "prop. of 0")
dimnames(XX)<- list(NULL, gsel)
pairs(XX, col=rainbow(mx, v=c(1,1,.5,.5), s=c(1,.5,1,.5))[cmod$cl], pch=(1:19)[cmod$cl %% 19 +1], cex.axis=1.5)
dev.off()
## more complex:


table(cmod$cl)

plot(table(cmod$cl))

idg<- which.max(table(cmod$cl))

idid<- which.min(as.matrix(dist(rbind(0,cmod$center)))[1,-1])

idid<- which.max(table(cmod$cl))
#idid<- 1

#library(foreach)
#library(doParallel)

#setup parallel backend to use many processors
#cores=detectCores()
#cl <- makeCluster(2) #not to overload your computer
#registerDoParallel(cl)

#outloop <- foreach(idid=1:mx) %dopar% {


for(idid in 0:mx){
  idsel <- which(idid == cmod$cl)
  idsel_itemid <- OO[idsel, "item_id"]
  idsel_is <- rep(idsel, n_stores) + rep((1:n_stores - 1) * n_item, rep(length(idsel), n_stores))
  idx <- rep(idp, rep(length(idsel_is), length(idp))) + rep(idsel_is, length(idp))
  
  idxxx <- idx[!is.na(train[idx, "sell_price"])]
  set.seed(1234)
  
  idout <- which(train[idxxx, d] == train[nrow(train), d])
  idxx_out <- idxxx[idout]
  idxxx_red <- idxxx[-idout]
  idxx_in <- sort(sample(idxxx_red, length(idxxx_red) * 1.0))
  idxx <- c(idxx_in, idxx_out)
  
  dat <- as.data.frame(train[idxx, ])
  
  tmp <- c(
    "demand", "item_id", "store_id", "d", "demand_lag_tH", "demand_lag_tHp1", "demand_lag_tHp6", "demand_lag_ann7",
    "demand_rolling_mean_t7", "demand_rolling_mean_t14", "demand_rolling_mean_t28",
    "demand_rolling_mean_t56", "demand_rolling_mean_ann7",
    
    "demand_store_id_lag_tH", "demand_store_id_lag_tHp1", "demand_store_id_lag_tHp6", "demand_store_id_lag_ann7",
    "demand_store_id_rolling_mean_t7", "demand_store_id_rolling_mean_t14",
    "demand_store_id_rolling_mean_t28", "demand_store_id_rolling_mean_t56",
    "demand_store_id_rolling_mean_ann7",
    
    "demand_item_id_lag_tH", "demand_item_id_lag_tHp1", "demand_item_id_lag_tHp6", "demand_item_id_lag_ann7",
    "demand_item_id_rolling_mean_t7", "demand_item_id_rolling_mean_t14",
    "demand_item_id_rolling_mean_t28", "demand_item_id_rolling_mean_t56",
    "demand_item_id_rolling_mean_ann7",
    
    "demand_all_id_lag_tH", "demand_all_id_lag_tHp1", "demand_all_id_lag_tHp6", "demand_all_id_lag_ann7",
    "demand_all_id_rolling_mean_t7", "demand_all_id_rolling_mean_t14",
    "demand_all_id_rolling_mean_t28", "demand_all_id_rolling_mean_t56",
    "demand_all_id_rolling_mean_ann7"
  )
  
  existing_cols <- intersect(tmp, names(dat))
  DAT <- dat[, existing_cols]
  DAT <- na.omit(DAT)  # Remove incomplete rows
  
  # Save to CSV
  cluster_filename <- paste0("/scratch/brussel/105/vsc10528/LSSboost/data/train_cluster_", idid, ".csv")
  fwrite(DAT, file = cluster_filename)
  
  cat("Saved cluster", idid, "\n")
}

for(idid in 81:mx){
idsel<- which(idid==cmod$cl)
iin<- idsel

idsel_itemid<- OO[idsel,"item_id"]
idsel_is <- rep(idsel, n_stores) + rep((1:n_stores-1)*n_item, rep(length(idsel),n_stores))
idx<-rep(idp, rep(length(idsel_is),length(idp)))  +rep( idsel_is, length(idp) )

	idxxx<- idx[ which(!is.na(train[idx,"sell_price"]) ) ]


set.seed(1234)
idout<- which(train[idxxx,d]== train[dim(train)[1],d])
idxx_out<- idxxx[idout]
idxxx_red<- idxxx[-idout]
idxx_sampling<- sample(idxxx_red,  length(idxxx_red)*1.0)
idxx_in<- sort(idxx_sampling)
idxx<- c(idxx_in,idxx_out)
	yy<- train[idxx, demand]
	dat<- as.data.frame(train[idxx,])


		tmp<- c(
"demand", "demand_lag_tH" ,  "demand_lag_tHp1","demand_lag_tHp6", "demand_lag_ann7" , "demand_rolling_mean_t7"  ,  "demand_rolling_mean_t14","demand_rolling_mean_t28","demand_rolling_mean_t56"  , "demand_rolling_mean_ann7"
,
"demand_store_id_lag_tH" ,  "demand_store_id_lag_tHp1", "demand_store_id_lag_tHp6", "demand_store_id_lag_ann7" , "demand_store_id_rolling_mean_t7"  ,  "demand_store_id_rolling_mean_t14","demand_store_id_rolling_mean_t28" ,"demand_store_id_rolling_mean_t56"    , "demand_store_id_rolling_mean_ann7"
,
"demand_item_id_lag_tH" ,  "demand_item_id_lag_tHp1","demand_item_id_lag_tHp6", "demand_item_id_lag_ann7" , "demand_item_id_rolling_mean_t7"  ,  "demand_item_id_rolling_mean_t14" ,  "demand_item_id_rolling_mean_t28" , "demand_item_id_rolling_mean_t56"    , "demand_item_id_rolling_mean_ann7"
,
"demand_all_id_lag_tH" ,  "demand_all_id_lag_tHp1","demand_all_id_lag_tHp6", "demand_all_id_lag_ann7" , "demand_all_id_rolling_mean_t7"  ,  "demand_all_id_rolling_mean_t14"  , "demand_all_id_rolling_mean_t28" ,"demand_all_id_rolling_mean_t56"  , "demand_all_id_rolling_mean_ann7"
) 


		DAT<- dat[, tmp]
		## extend by sum across I
#		DAT

		idDAT<- which(  !apply(is.na(DAT),1,any)  ) #&  dat[, "d"]> max(dat[, "d"])-ptmax  )#!is.na(DAT[,"rolling_mean_ann7"])#!apply(is.na(DAT),1,any)
#		DAT<- DAT[idDAT,]

#		system.time( mod<- gam(demand~ rolling_mean_t7 + store_id + te(wday) , family="nb", data=DAT ) )

		#WDAY<- dat[, "wday"] == matrix(c(1,5:7), dim(dat)[1],4, byrow=TRUE)
		#dimnames(WDAY)<- list(NULL, paste("wday", c(1,5:7), sep=""))
		#STOREID<-  dat[, "store_id"] == matrix(1:n_stores, dim(dat)[1],n_stores, byrow=TRUE)
		#dimnames(STOREID)<- list(NULL, paste("store", 1:n_stores, sep=""))
		#ITEMID<-  dat[, "item_id"] == matrix(idsel_itemid, dim(dat)[1],length(idsel), byrow=TRUE)
		#dimnames(ITEMID)<- list(NULL, paste("item", idsel_itemid, sep=""))
		##interactions
#gc()

		#DAT<- data.table(cbind( dat[, tmp],  WDAY , STOREID , ITEMID))#[idDAT,] ## wday:1== Sat
		DAT <- dat[, tmp]
		names(DAT)<- gsub("TRUE","",names(DAT))

		cluster_filename <- paste0("/scratch/brussel/105/vsc10528/LSSboost/data/train_cluster_", idid, ".csv")
		fwrite(DAT, file = cluster_filename)
}
#sort( sapply(ls(),function(x){object.size(get(x))})) 
gc()


library(gamlss.lasso)
		idxvar<- which(names(DAT)!="demand")

		MODL<- c("PO", "GEOM", "NBI", "WARING", "DPO","GPO","ZIP")#, "BNB", "PIG", "ZINBI")
		i.dist<- 7
for(i.dist in seq_along(MODL)){
#for(i.dist in 5:7){
		dfamily<- MODL[i.dist]
		kmax<- length(formals(get(dfamily)))

		act.cyc<- 5

#		system.time(mod<- gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family=dfamily, data=DAT[idDAT,] , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = act.cyc)) ) ##
		mod<- NULL
		while(act.cyc>=1 & is.null(mod) ){
		mod<- tryCatch({gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family=dfamily, data=DAT[idDAT,] , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = act.cyc))},
			error=function(e) NULL)
			act.cyc<- act.cyc-1 #retry with one iteration less - not optimal in WC 3times more effort for n.cyc.init=5
		}
		if(is.null(mod)) next ##should never happen		

		QQ<- c(0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995)
## in-sample
#		pmu<- predict(mod, what="mu", type="response")
#		psigma<- predict(mod, what="sigma", type="response")
#	#	pnu<- predict(mod, what="nu", type="response")

#		QQp<- matrix(QQ, dim(DAT)[1],length(QQ), byrow=TRUE)
#		PQ<- QQp
#		for(i.q in seq_along(QQ)) PQ[,i.q]<- eval(parse(text= paste(paste("q",dfamily, sep=""), "(QQp[,i.q], pmu, psigma)",sep="")) )#, takes quite long...

IC<- AIC(mod, k=c(0,2, 2*log(log(length(idDAT))), log(length(idDAT))) )/length(idDAT)
names(IC)<- c("-2loglik", "AIC", "HQC", "BIC")

DATout<- tail(DAT[,-1], length(idsel)*n_stores)
DAToutm<- Matrix(as.matrix(tail(DAT[,-1], length(idsel)*n_stores)), sparse=TRUE)
QQp<- matrix(QQ, dim(DAToutm)[1],length(QQ), byrow=TRUE)
PQpred<- QQp

if(kmax==2){
# two-parametric
ggm<- getSmo(mod, "mu")[[2]] 
ggs<- getSmo(mod, "sigma")[[2]] 
if(dfamily=="ZIP"){
pred_par<- cbind( "mu" = exp( DAToutm %*% ggm$beta + ggm$a0 + coef(mod, "mu")[1] ), "sigma"= 1/(1+ exp( -( DAToutm %*% ggs$beta + ggs$a0 + coef(mod, "sigma")[1] )) ) )
} else { #log-log case
pred_par<- cbind( "mu" = exp( DAToutm %*% ggm$beta + ggm$a0 + coef(mod, "mu")[1] ), "sigma"= exp( DAToutm %*% ggs$beta + ggs$a0 + coef(mod, "sigma")[1] ) )
}
for(i.q in seq_along(QQ)) PQpred[,i.q]<- eval(parse(text= paste(paste("q",dfamily, sep=""), "(QQp[,i.q], pred_par[,1], pred_par[,2])",sep="")) )#, pnu)
} else {
ggm<- getSmo(mod, "mu")[[2]] 
pred_par<- cbind( "mu" = exp( DAToutm %*% ggm$beta + ggm$a0 + coef(mod, "mu")[1] ) )
for(i.q in seq_along(QQ)) PQpred[,i.q]<- eval(parse(text= paste(paste("q",dfamily, sep=""), "(QQp[,i.q], pred_par[,1])",sep="")) )#, pnu)
}

#oo_item_id<- rep( idsel_itemid, n_stores)
#oo_store_id<- rep( 1:n_stores, rep(length(idsel_itemid) , n_stores))
dimnames(PQpred)<- list( NULL,paste("q",QQ*100, sep=""))

OUT<- cbind( train[idxx_out, c("item_id", "store_id")] , PQpred) 

bbpath<- "~/sciebo/PRIVATE/m5/proposal/out_rev/"
dir.create( paste(bbpath, "/",dfamily,sep="") )
dir.create( paste(bbpath, "/",dfamily, "/",H,sep="") )
write.csv(OUT, paste(bbpath, "/",dfamily, "/",H,"/",idid, ".csv",sep=""), row.names=FALSE, quote=FALSE)
write.csv(t(IC), paste(bbpath, "/",dfamily, "/",H,"/IC_",idid, ".csv",sep=""), row.names=FALSE, quote=FALSE)

#		forecast<- tryCatch({forecast.gnet.gamlss(dat, days, pgrid, dfamily=as.character(model.list[i.m,1]), kpar=1:as.numeric(model.list[i.m,2]), lambda=model.list[i.m,3][[1]]  ,adaptive=model.list[i.m,4][[1]])},
#		error=function(e)list(f=array(, dim=c(S, P)),ic=array(, dim=c(S, 3))))


par(mfrow=c(2,1))
plot(getSmo(mod, "mu")[[2]]$scaled.beta, type="h", main=dfamily)
if(kmax>1)plot(getSmo(mod, "sigma")[[2]]$scaled.beta, type="h")
par(mfrow=c(1,1))
gc()

}
		
#i.dist
gc()
	NULL
}#ii

#stopCluster(cl)


system.time({
		mod<- tryCatch({gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family=dfamily, data=DAT[idDAT,] , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 1))},
			error=function(e) NULL)
})

system.time({
		mod<- tryCatch({gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family=dfamily, data=DAT[idDAT,] , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 2))},
			error=function(e) NULL)
})

system.time({
		mod<- tryCatch({gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family=dfamily, data=DAT[idDAT,] , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 1))},
			error=function(e) NULL)
})

	update(mod, control=gamlss.control(n.cyc = 3) )
	
	
		system.time(mod<- gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family="PO", data=DAT[idDAT,] , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 5)) )

		system.time(mod<- gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family="GEOM", data=DAT[idDAT,] , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 5)) )

		system.time(mod<- gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family="NBI", data=DAT[idDAT,] , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 5)) )

		system.time(mod<- gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE), nu.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family="WARING", data=DAT[idDAT,] , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 5)) ) ##

		system.time(mod<- gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family="PIG", data=DAT[idDAT,] , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 5)) ) ## error

		system.time(mod<- gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE), nu.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family="BNB", data=DAT[idDAT,] , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 5)) ) ##

#		system.time(mod<- gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE), nu.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family="ZINBI", data=DAT , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 25)) ) ##

#		system.time(mod<- gamlss(demand~ . ,sigma.fo=~ .,  family="NBI", data=DAT , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 5)) )

par(mfrow=c(3,1))
plot(getSmo(mod, "mu")[[2]]$beta, type="h")
plot(getSmo(mod, "sigma")[[2]]$beta, type="h")
plot(getSmo(mod, "nu")[[2]]$beta, type="h")
par(mfrow=c(1,1))

		pmu<- predict(mod, what="mu", type="response")
		psigma<- predict(mod, what="sigma", type="response")
	#	pnu<- predict(mod, what="nu", type="response")
		QQ<- c(0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995)
		QQp<- matrix(QQ, dim(DAT)[1],length(QQ), byrow=TRUE)
		PQ<- QQp
		for(i.q in seq_along(QQ)) PQ[,i.q]<- qNBI(QQp[,i.q], pmu, psigma)#, pnu)

DATout<- tail(DAT[,-1], length(idsel)*n_stores)
#pred_DAT<- predictAll(mod, newdata=DATout, type="response")

DAToutm<- Matrix(as.matrix(tail(DAT[,-1], length(idsel)*n_stores)), sparse=TRUE)

ggm<- getSmo(mod, "mu")[[2]] 
ggs<- getSmo(mod, "sigma")[[2]] 
pred_par<- cbind( "mu" = exp( DAToutm %*% ggm$beta + ggm$a0 + coef(mod, "mu")[1] ), "sigma"= exp( DAToutm %*% ggs$beta + ggs$a0 + coef(mod, "sigma")[1] ) )

dfamily="NBI"

		QQp<- matrix(QQ, dim(DAToutm)[1],length(QQ), byrow=TRUE)
		PQpred<- QQp
		for(i.q in seq_along(QQ)) PQpred[,i.q]<- eval(parse(text= paste(paste("q",dfamily, sep=""), "(QQp[,i.q], pred_par[,1], pred_par[,2])",sep="")) )#, pnu)




DAT[,y]

#358457.8

#		system.time(mod<- gamlss(demand~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE) ,sigma.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),nu.fo=~ gnet(x.vars=idxvar,ICpen="HQC", sparse=TRUE),  family="ZINBI", data=DAT , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 5)) )

#par(mfrow=c(3,1))
#plot(getSmo(mod, "mu")[[2]]$beta, type="h")
#plot(getSmo(mod, "sigma")[[2]]$beta, type="h")
#plot(getSmo(mod, "nu")[[2]]$beta, type="h")
#par(mfrow=c(1,1))

#		pmu<- predict(mod, what="mu", type="response")
#		psigma<- predict(mod, what="sigma", type="response")
#		pnu<- predict(mod, what="nu", type="response")
#		QQ<- c(0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995)
#		QQp<- matrix(QQ, dim(DAT)[1],length(QQ), byrow=TRUE)
#		PQ<- QQp
#		for(i.q in seq_along(QQ)) PQ[,i.q]<- qZINBI(QQp[,i.q], pmu, psigma)#, pnu)


		#
#> dim(DAT)
#[1] 87383    82


		ii.item<- 1
		ii.store<- 1
#		pdf(paste("/home/florian/sciebo/PRIVATE/m5/proposal/out/gamlss01_items.pdf", sep=""), width=8, height=5)
	
		for(ii.store in 1:n_stores)for(ii.item in seq_along(iitem)){
		pidx<- which( dat[idDAT, "item_id"] == idsel_itemid[ii.item] & dat[idDAT, "store_id"] == ii.store)

head(dat[idDAT,1:12],740)

#		plot(DAT[pidx,demand], type="o")
		COL<- rev(rainbow(length(QQ), v=c(1,1,1,1, 0.6, 1,1,1,1)))
		tdat<- dat[idDAT, "date"][pidx]
		PPQ<- PQ[pidx,]
		xlims<- tail(tdat, ptmax)

#		pdf(paste("/home/florian/sciebo/PRIVATE/m5/proposal/out/gamlss1_item",10000+(ii.item-1)*n_stores+ii.store, ".pdf", sep=""), width=8, height=5)
		par(family="Times", mar=c(4.1,4.3,.6,.2))
		plot(tdat, PPQ[,length(QQ)], col="white", ylim=range(c(0, tail(PPQ[,length(QQ)],ptmax), tail(DAT[pidx,demand],ptmax)))*c(1,1.15), las=1, ylab=paste("Demand, item=",iitem[ii.item],", store=",ii.store ,sep=""), xlab="Time", xlim=as.numeric(range(xlims)), xaxt="n", cex.lab=1.3, cex.axis=1.3)
		grid(nx=NA, ny=NULL)
		pts<- tail(tdat, ptmax)[substr(tail(tdat, ptmax), 6,10) %in% c("01-01","04-01","07-01","10-01")]
		axis(1, pts, format(pts, "%b'%y"), cex.axis=1.3)
#which.max(!apply(PPQ==0,2,all))
		for(i.q in seq_along(QQ)) lines(tdat, PPQ[,i.q], col=COL[i.q])

		lines(tdat,DAT[pidx,demand])
		legend("topleft", c("data", paste("q",QQ*100,"%", sep="") ) , col=c(1,COL),bg=rgb(1,1,1,.8), ncol=5, lwd=2, cex=1.2)
#		dev.off()
		}
#		dev.off()

## INTERACTION MODEL

		DATm<- as.matrix(DAT)
		DATT<- as.data.frame(array(, dim=c(dim(DAT)[1], dim(DAT)[2] +(dim(DAT)[2]-2)*(dim(DAT)[2]-1)/2 )  ))
		DATT[, 1:dim(DAT)[2]]<- DATm
		k<- dim(DAT)[2]
		kp<- dim(DAT)[2]
		for(i in 1:(dim(DAT)[2]-2)) {
#			ij<- i+1
#			subs<- (ij+1):kp 
			DATT[,k+1:(kp-i-2)]<- DATm[,i+1] * DATm[, (i+2):kp ]
			k<- k+ (kp-i-1)
		}
		form <- demand ~ .^2
		dnames<- gsub("TRUE","" ,dimnames(model.matrix(form, data = tail(DAT,1)))[[2]])
		names(DATT)<- c(names(DAT),dnames[-c(1:k)])
		tmp<-apply(DATT,2, sum, na.rm=TRUE)
		DATT<- DATT[,tmp!=0]
		
#		system.time(DATTT<- data.table(DATT)) #data.table(DATT[,tmp!=0]) #as.data.frame
#		DATT<- cbind(DATT, demand=unlist(DAT[,"demand"]))
#		names(DATT)<- gsub("TRUE","",names(DATT))
#		gc()
#		idDAT<- !apply(is.na(DAT),1,any)
#		DAT<- DAT[idDAT,]
	#	system.time(mod<- gamlss(demand~ . , family="NBI", data=DAT ))
	#	system.time(mod<- gamlss(demand~ . ,sigma.fo=~ .,  family="NBI", data=DAT ))
		dim(DATT)

		system.time(mod<- gamlss(demand~ gnet(x.vars=which(names(DATT)!="demand"),ICpen="BIC", sparse=TRUE, control = gnet.control(pmax=100) ) ,sigma.fo=~ gnet(x.vars=which(names(DATT)!="demand"),ICpen="BIC", sparse=TRUE, control = gnet.control(pmax=100)),  family="NBI", data=DATT[idDAT,] , i.control = glim.control(cyc=1, bf.cyc=1), control=gamlss.control(n.cyc = 1)) )
#[sample(seq_along(idDAT),20000)]
par(mfrow=c(3,1))
plot(getSmo(mod, "mu")[[2]]$beta, type="h")
plot(getSmo(mod, "sigma")[[2]]$beta, type="h")
plot(getSmo(mod, "nu")[[2]]$beta, type="h")
par(mfrow=c(1,1))


		pmu<- predict(mod, what="mu", type="response")
		psigma<- predict(mod, what="sigma", type="response")
		QQ<- c(0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995)
		QQp<- matrix(QQ, dim(DAT)[1],length(QQ), byrow=TRUE)
		PQ<- QQp
		for(i.q in seq_along(QQ)) {
			PQ[,i.q]<- qNBI(QQp[,i.q], pmu, psigma)
			cat(i.q)
		}

		ii.item<- 1
		ii.store<- 1
		pdf(paste("/home/florian/sciebo/PRIVATE/m5/proposal/out/gamlss02_items.pdf", sep=""), width=8, height=5)
	
		for(ii.store in 1:n_stores)for(ii.item in seq_along(iitem)){
		pidx<- which( dat[idDAT, "item_id"] == iitem[ii.item] & dat[idDAT, "store_id"] == ii.store)

#		plot(DAT[pidx,demand], type="o")
		COL<- rev(rainbow(length(QQ), v=c(1,1,1,1, 0.6, 1,1,1,1)))
		tdat<- dat[idDAT, "date"][pidx]
		PPQ<- PQ[pidx,]
		ptmax<- 730
		xlims<- tail(tdat, ptmax)

#		pdf(paste("/home/florian/sciebo/PRIVATE/m5/proposal/out/gamlss02_item",10000+(ii.item-1)*n_stores+ii.store, ".pdf", sep=""), width=8, height=5)
		par(family="Times", mar=c(4.1,4.3,.6,.2))
		plot(tdat, PPQ[,length(QQ)], col="white", ylim=range(c(0, tail(PPQ[,length(QQ)],ptmax), tail(DAT[pidx,demand],ptmax)))*c(1,1.15), las=1, ylab=paste("Demand, item=",iitem[ii.item],", store=",ii.store ,sep=""), xlab="Time", xlim=as.numeric(range(xlims)), xaxt="n", cex.lab=1.3, cex.axis=1.3)
		grid(nx=NA, ny=NULL)
		pts<- tail(tdat, ptmax)[substr(tail(tdat, ptmax), 6,10) %in% c("01-01","04-01","07-01","10-01")]
		axis(1, pts, format(pts, "%b'%y"), cex.axis=1.3)
#which.max(!apply(PPQ==0,2,all))
		for(i.q in seq_along(QQ)) lines(tdat, PPQ[,i.q], col=COL[i.q])

		lines(tdat,DAT[pidx,demand])
		legend("topleft", c("data", paste("q",QQ*100,"%", sep="") ) , col=c(1,COL),bg=rgb(1,1,1,.8), ncol=5, lwd=2, cex=1.2)
	#	dev.off()
		}
		dev.off()


sort( sapply(ls(),function(x){object.size(get(x))})) 

par(mfrow=c(3,1))
plot(getSmo(mod, "mu")[[2]]$beta, type="h")
plot(getSmo(mod, "sigma")[[2]]$beta, type="h")
plot(getSmo(mod, "nu")[[2]]$beta, type="h")
par(mfrow=c(1,1))

#
		system.time(mod<- gamlss(demand~ . , family="NBI", data=DAT ))


		summary(mod<- gam(demand~ rolling_mean_t7 + te(wday) , family="nb", data=DAT ))
		summary(mod<- bam(demand~ rolling_mean_t7 + te(wday) , family="nb", data=DAT ))

#		mod<- gamlss(demand~ rolling_mean_t7 + te(wday) , family="NBI", control=gamlss.control(trace=FALSE), data=DAT )

	


		}, error = function(e) {})
		if(!is.null(mod)){
#				qs <-summary(mod, "qr")
				summaryoutput<-capture.output(qs<-summary(mod, "qr"))
				k<- mod$df.fit #dim(qs)[1]
				if(dim(DAT)[2]==1){
					cname<- c("mu", "sigma", "nu", "tau")[1:k]
					modlink<- unlist(mod[paste(mod$par,".link", sep="")])
					dgg<- 0:4
				
					if(tail(modlink,1)=="logit"){ ## zero-infl. case
						cname[length(cname)]<- "z0"
						ctmp<- c( sapply( as.numeric(coefAll(mod)[-k]), exp), pmax( 1/(1+exp(-as.numeric(coefAll(mod)[k])) ), 1e-16)			)	
					} else {
						ctmp<- sapply( coefAll(mod), exp)
		#				OM[imod,ii, cname[1:k]]<- ctmp[1:k]
					}	
					names(ctmp)<- cname[1:k]
					OM[imod,ix,ii, cname[1:k]]<- ctmp[1:k]
					OM[imod,ix,ii, paste(cname[1:k], ".tval",sep="")]<- qs[,3]
					OM[imod,ix,ii, paste(cname[1:k], ".pval",sep="")]<- qs[,4]
					if(k==1) tryCatch({OM[imod,ix,ii, paste("d", dgg,sep="")]<- get(paste("d",MODL[[imod]], sep=""))(dgg, ctmp[1:k])}, error = function(e) {})
					if(k==2) tryCatch({OM[imod,ix,ii, paste("d", dgg,sep="")]<- get(paste("d",MODL[[imod]], sep=""))(dgg, ctmp[1],ctmp[2])}, error = function(e) {})
					if(k==3) tryCatch({OM[imod,ix,ii, paste("d", dgg,sep="")]<- get(paste("d",MODL[[imod]], sep=""))(dgg, ctmp[1],ctmp[2],ctmp[3])}, error = function(e) {})
					if(k==4) tryCatch({OM[imod,ix,ii, paste("d", dgg,sep="")]<- get(paste("d",MODL[[imod]], sep=""))(dgg, ctmp[1],ctmp[2],ctmp[3],ctmp[4])}, error = function(e) {})
				}## simple models
				OM[imod,ix,ii,"AIC"]<- AIC(mod)
				OM[imod,ix,ii,"BIC"]<- BIC(mod)
			}#null mod
		}#ix
		cat(imod, ".")
	}#imod

	print(ii)
}#iitem






