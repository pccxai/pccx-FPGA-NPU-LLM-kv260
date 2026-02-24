# 1. Structure and connection
```mermaid
graph LR
    %% 입력 정의
    InA0[("In A_0")] --> PE00
    InB0[("In B_0")] --> PE00
    InA1[("In A_1")] --> PE10
    InB1[("In B_1")] --> PE01

    %% PE 배치 및 연결
    subgraph "Systolic Array 2x2"
        direction TB
        PE00[PE 0,0] --"a(delay)"--> PE01[PE 0,1]
        PE00 --"b(delay)"--> PE10[PE 1,0]
        
        PE01 --"b(delay)"--> PE11[PE 1,1]
        PE10 --"a(delay)"--> PE11
    end

    %% 스타일링
    classDef pe fill:#fff,stroke:#333,stroke-width:2px;
    class PE00,PE01,PE10,PE11 pe;
```

# 2. Step-by-Step Execution

<ul> <h3>Cycle 1: 첫 번째 파동 (Start)    </h3>  
<li>Input: in_a_0=1, in_b_0=1  (in_a_1, in_b_1 are yet 0)  </li>
<li>Run: only **PE(0,0)** begin operation. other PE's wait for signal</li>
</li>
</ul>

```mermaid
---
title: Cycle 1 - Activate PE(0,0)
---
graph TD
    subgraph Inputs
    A0["A0=1"] --> PE00
    B0["B0=1"] --> PE00
    A1["A1=0 (대기)"] -.-> PE10
    B1["B1=0 (대기)"] -.-> PE01
    end

    subgraph Array
    PE00("PE 0,0<br/><b>1 * 1 = 1</b><br/>(Acc: 1)")
    PE01("PE 0,1<br/>Idle")
    PE10("PE 1,0<br/>Idle")
    PE11("PE 1,1<br/>Idle")
    
    PE00 -->|Next: a=1| PE01
    PE00 -->|Next: b=1| PE10
    PE01 -.-> PE11
    PE10 -.-> PE11
    end

    classDef active fill:#ff9,stroke:#f66,stroke-width:4px;
    classDef idle fill:#eee,stroke:#999,stroke-dasharray: 5 5;
    class PE00 active;
    class PE01,PE10,PE11 idle;
```

<ul> <h3>Cycle 2: 확산 (Propagation)  </h3>  
<li>Input: in_a_0=2, in_b_0=3 (두 번째 데이터), in_a_1=3, in_b_1=2 (첫 번째 데이터가 지연되어 도착)  </li>
<li>Pass Data: PE(0,0)이 아까 쓴 a=1, b=1을 각각 오른쪽(PE01)과 아래쪽(PE10)으로 넘겨줍니다.</li>
<li>Run: 이제 PE(0,0)(2차 연산), PE(0,1), PE(1,0) 세 곳에서 동시에 불이 켜집니다.
</li>
</ul>

```mermaid
---
title: Cycle 2 - Wavefront data
---
graph TD

    subgraph Inputs
    A0["A0=2"] --> PE00
    B0["B0=3"] --> PE00
    A1["A1=3"] --> PE10
    B1["B1=2"] --> PE01
    end

    subgraph Array
    PE00("PE 0,0<br/><b>2 * 3 = 6</b><br/>(Acc: 1+6=7)")
    
    PE01("PE 0,1<br/>From Left: 1<br/>From Top: 2<br/><b>1 * 2 = 2</b><br/>(Acc: 2)")
    
    PE10("PE 1,0<br/>From Left: 3<br/>From Top: 1<br/><b>3 * 1 = 3</b><br/>(Acc: 3)")
    
    PE11("PE 1,1<br/>Idle")
    
    PE00 -->|Next: a=2| PE01
    PE00 -->|Next: b=3| PE10
    PE01 -->|Next: b=2| PE11
    PE10 -->|Next: a=3| PE11
    end

    classDef active fill:#ff9,stroke:#f66,stroke-width:4px;
    classDef idle fill:#eee,stroke:#999,stroke-dasharray: 5 5;
    class PE00,PE01,PE10 active;
    class PE11 idle;
```
<ul> <h3> Cycle 3: 수렴 (Convergence)  </h3>  
<li>입력: in_a_1=4, in_b_1=4 (마지막 데이터)</li>
<li>전달: PE(0,1)과 PE(1,0)이 처리한 데이터들이 **PE(1,1)**로 모입니다.</li>
<li>동작: 드디어 **PE(1,1)**이 첫 연산을 시작합니다. 나머지 PE들도 계속 연산을 수행합니다.
</li>
</ul>

```mermaid
---
title: Cycle 3 - PE(1,1) 도달 및 전체 가동
---
graph TD

    subgraph Inputs
    A0["A0=0"] --> PE00
    B0["B0=0"] --> PE00
    A1["A1=4"] --> PE10
    B1["B1=4"] --> PE01
    end

    subgraph Array
    PE00("PE 0,0<br/>Idle")
    
    PE01("PE 0,1<br/>From Left: 2<br/>From Top: 4<br/><b>2 * 4 = 8</b><br/>(Acc: 2+8=10)")
    
    PE10("PE 1,0<br/>From Left: 4<br/>From Top: 3<br/><b>4 * 3 = 12</b><br/>(Acc: 3+12=15)")
    
    PE11("PE 1,1<br/>From Left: 3<br/>From Top: 2<br/><b>3 * 2 = 6</b><br/>(Acc: 6)")
    
    PE00 -.-> PE01
    PE00 -.-> PE10
    PE01 -->|Next: b=4| PE11
    PE10 -->|Next: a=4| PE11
    end

    classDef active fill:#ff9,stroke:#f66,stroke-width:4px;
    classDef idle fill:#eee,stroke:#999,stroke-dasharray: 5 5;
    class PE01,PE10,PE11 active;
    class PE00 idle;
```  

---

<ul> <h3>Cycle 4: 마무리 (Tail) </h3>  
<li>입력: 데이터 주입 끝 (Valid Off).</li>
<li>동작: 파이프라인에 남아있는 마지막 데이터들이 **PE(1,1)**에서 처리됩니다.</li>
</ul>

```mermaid
---
title: Cycle 4 - 마지막 데이터 처리
---
graph TD
    subgraph Array
    PE00("PE 0,0<br/>Done")
    PE01("PE 0,1<br/>Done")
    PE10("PE 1,0<br/>Done")
    
    PE11("PE 1,1<br/>From Left: 4<br/>From Top: 4<br/><b>4 * 4 = 16</b><br/>(Acc: 6+16=22)")
    
    PE00 -.-> PE01
    PE00 -.-> PE10
    PE01 -.-> PE11
    PE10 -.-> PE11
    end

    classDef active fill:#ff9,stroke:#f66,stroke-width:4px;
    classDef done fill:#cfc,stroke:#333,stroke-width:2px;
    class PE11 active;
    class PE00,PE01,PE10 done;
```

# Summary
1. Cycle 1: (0,0) start
2. Cycle 2: (0,0) second operation  + (0,1), (1,0) first operation (data pass)
3. Cycle 3: (0,1), (1,0) second operation + (1,1) first operation
4. Cycle 4: (1,1) second operation