s0 = 421017
i0 = 5044/(7.4*(10**5))
r0 = i0

total_time = 250
time_step = 1 

st = [s0/s0]
it = [i0/s0]
rt = [r0/s0]
b = 0.178
k = 0.015
## Traditional SIR Model

for time in range(0, total_time, time_step):
    t = int(time/time_step)

    ds = -b*st[t]*it[t]
    
    s1 = st[t] + ds
    st.append(s1)

    di = b*st[t]*it[t] - k*it[t]
    i1 = it[t] + di
    it.append(i1)

    dr = k*it[t]
    r1 = rt[t] + dr
    rt.append(r1)

## Quick Plot 
times = [t for t in range(0, total_time+time_step, time_step)]
print(it[1])
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv(r'E:\Research Project\Python Models\COVID19_Data_2020\Berks.csv')

## Add a new column: New Confirmed Cases and New Deaths (these will represent X(1), X(2),..., X(n))
total_confirmed = list(df['Confirmed']) # The data from the CDC is strictly the total (Tn) number of conformed cases
Tn = list(df['Confirmed'])
generation_time = 5
Tn = Tn[0::generation_time]

Tn = [t/s0 for t in Tn]

# plot lines 
plt.scatter(range(0+80, len(Tn)+80), Tn)
plt.plot(times, st, label = "S(t)")
plt.plot(times, it, label = "I(t)") 
plt.plot(times, rt, label = "R(t)") 
plt.title("Example SIR Data")
plt.xlabel("Time (n)")
plt.ylabel("Proportion")
plt.legend() 
plt.show()